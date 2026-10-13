from scripts.drive_sync import BEST_FILENAME, DRIVE_FRCRN_DIR
from scripts.load_pretrain import download_frcrn_best


class FakeDrive:
    def __init__(self):
        self.gets = []

    def get_file(self, remote, local):
        self.gets.append((remote, local))
        with open(local, "wb") as f:
            f.write(b"ckpt")


def test_download_frcrn_best_fetches_best_from_frcrn_folder(tmp_path):
    drive = FakeDrive()
    destination = tmp_path / "pretrained" / "frcrn_best.pt"

    result = download_frcrn_best(drive, destination)

    assert result == destination
    assert drive.gets == [(f"{DRIVE_FRCRN_DIR}/{BEST_FILENAME}", str(destination))]
    assert destination.read_bytes() == b"ckpt"


def test_download_frcrn_best_skips_existing_file(tmp_path):
    drive = FakeDrive()
    destination = tmp_path / "frcrn_best.pt"
    destination.write_bytes(b"old")

    download_frcrn_best(drive, destination)

    assert drive.gets == []
    assert destination.read_bytes() == b"old"
