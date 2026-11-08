import unittest
from io import BytesIO
import base64

import numpy as np
from PIL import Image

from classifier import SatelliteClassifier
from paste_input import decode_clipboard_payload
from image_processing import (
    class_percentages,
    combine_masks,
    class_transition_counts,
    compare_landcover_masks,
    create_tiles,
    decode_uploaded_image,
    render_classification,
)
from parallel_processing import run_batched, run_sequential


class ImagePipelineTests(unittest.TestCase):
    def test_tiling_stitches_to_original_dimensions(self) -> None:
        image = np.zeros((79, 130, 3), dtype=np.uint8)
        tiles = create_tiles(image, 64)
        masks = [
            np.full((tile.height, tile.width), index % 4, dtype=np.uint8)
            for index, tile in enumerate(tiles)
        ]

        result = combine_masks(image.shape, tiles, masks)
        self.assertEqual(len(tiles), 6)
        self.assertEqual(result.shape, image.shape[:2])
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[0, 64], 1)
        self.assertAlmostEqual(sum(class_percentages(result)), 100.0)

    def test_sequential_and_batched_paths_match(self) -> None:
        image = np.zeros((40, 48, 3), dtype=np.uint8)
        image[:, :, 1] = 170
        image[10:30, 20:40, 2] = 230
        tiles = create_tiles(image, 16)
        classifier = SatelliteClassifier()

        sequential = run_sequential(classifier, tiles)
        batched = run_batched(classifier, tiles, batch_size=3)

        self.assertEqual(len(sequential.masks), len(tiles))
        self.assertEqual(len(batched.masks), len(tiles))
        for sequential_mask, batched_mask in zip(sequential.masks, batched.masks):
            np.testing.assert_array_equal(sequential_mask, batched_mask)

    def test_demo_classifier_identifies_representative_green_and_water_colors(self) -> None:
        classifier = SatelliteClassifier(device="cpu")
        samples = {
            "dark_green": ((30, 75, 35), 1),
            "vegetation": ((45, 130, 55), 1),
            "blue_water": ((35, 95, 180), 2),
            "dark_water": ((25, 55, 75), 2),
            "cyan_water": ((70, 125, 155), 2),
            "gray_urban": ((145, 145, 145), 3),
            "brown_land": ((145, 105, 65), 0),
            "sand": ((190, 170, 125), 0),
        }
        for label, (rgb, expected_id) in samples.items():
            with self.subTest(label=label):
                tile = np.full((16, 16, 3), rgb, dtype=np.uint8)
                prediction = classifier.predict_batch([tile])[0]
                self.assertTrue(np.all(prediction == expected_id))

    def test_upload_validation_checks_real_image_content(self) -> None:
        source = np.full((12, 15, 3), 127, dtype=np.uint8)
        buffer = BytesIO()
        Image.fromarray(source).save(buffer, format="PNG")

        decoded = decode_uploaded_image(buffer.getvalue(), "image.png")
        np.testing.assert_array_equal(decoded, source)
        with self.assertRaises(ValueError):
            decode_uploaded_image(b"not an image", "image.png")

    def test_paste_validation_rejects_vector_content(self) -> None:
        with self.assertRaises(ValueError):
            decode_uploaded_image(
                b"<svg xmlns='http://www.w3.org/2000/svg'></svg>",
                "clipboard.svg",
            )

    def test_paste_decoder_accepts_common_raster_formats(self) -> None:
        source = np.full((12, 15, 3), 127, dtype=np.uint8)
        for image_format in ("WEBP", "BMP", "TIFF", "GIF"):
            with self.subTest(image_format=image_format):
                buffer = BytesIO()
                Image.fromarray(source).save(buffer, format=image_format)
                decoded = decode_uploaded_image(
                    buffer.getvalue(), f"clipboard.{image_format.lower()}"
                )
                self.assertEqual(decoded.shape, source.shape)

    def test_clipboard_payload_decodes_image_bytes(self) -> None:
        image_buffer = BytesIO()
        Image.fromarray(np.full((4, 5, 3), 80, dtype=np.uint8)).save(
            image_buffer, format="PNG"
        )
        payload = {
            "data": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
            "mime": "image/png",
            "name": "clipboard image",
            "size": len(image_buffer.getvalue()),
        }

        decoded_payload = decode_clipboard_payload(payload, "Before image")
        self.assertIsNotNone(decoded_payload)
        image_bytes, name = decoded_payload
        decoded = decode_uploaded_image(image_bytes, name)
        self.assertEqual(decoded.shape, (4, 5, 3))

    def test_clipboard_payload_rejects_non_image_mime(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an image"):
            decode_clipboard_payload(
                {"data": "aGVsbG8=", "mime": "text/plain"},
                "Before image",
            )

    def test_map_can_filter_classes_without_changing_original(self) -> None:
        image = np.full((2, 2, 3), 100, dtype=np.uint8)
        mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)

        color_map, overlay = render_classification(
            image, mask, alpha=0.5, visible_classes={0, 2}
        )

        self.assertEqual(color_map[0, 0].tolist(), [255, 220, 0])
        self.assertEqual(color_map[0, 1].tolist(), [35, 42, 52])
        self.assertEqual(color_map[1, 0].tolist(), [35, 115, 235])
        self.assertEqual(color_map[1, 1].tolist(), [35, 42, 52])
        self.assertFalse(np.array_equal(overlay, image))

    def test_change_detection_identifies_water_gains_and_losses(self) -> None:
        before = np.array([[0, 2, 1], [3, 0, 2]], dtype=np.uint8)
        after = np.array([[2, 2, 0], [3, 1, 0]], dtype=np.uint8)

        changed, water_gain, water_loss = compare_landcover_masks(before, after)

        self.assertEqual(int(changed.sum()), 4)
        self.assertEqual(int(water_gain.sum()), 1)
        self.assertEqual(int(water_loss.sum()), 1)
        self.assertTrue(water_gain[0, 0])
        self.assertTrue(water_loss[1, 2])

    def test_change_detection_rejects_unaligned_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            compare_landcover_masks(
                np.zeros((2, 3), dtype=np.uint8),
                np.zeros((3, 2), dtype=np.uint8),
            )

    def test_transition_matrix_reports_each_before_after_class_count(self) -> None:
        before = np.array([[0, 0, 1], [2, 2, 3]], dtype=np.uint8)
        after = np.array([[0, 2, 1], [3, 2, 0]], dtype=np.uint8)

        transitions = class_transition_counts(before, after)

        self.assertEqual(transitions.shape, (4, 4))
        self.assertEqual(int(transitions.sum()), before.size)
        self.assertEqual(transitions[0, 0], 1)
        self.assertEqual(transitions[0, 2], 1)
        self.assertEqual(transitions[2, 3], 1)
        self.assertEqual(transitions[3, 0], 1)

if __name__ == "__main__":
    unittest.main()
