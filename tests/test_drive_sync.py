import logging
import time

from scripts.drive_sync import (
    BEST_FILENAME,
    DRIVE_FRCRN_VLSP_DIR,
    DriveCheckpointUploader,
    DriveFileBackup,
    DriveLogUploader,
)


class FakeDrive:
    def __init__(self, fail=False):
        self.fail = fail
        self.made = []
        self.puts = []

    def makedirs(self, path, exist_ok=False):
        self.made.append((path, exist_ok))

    def put_file(self, local, remote):
        if self.fail:
            raise OSError("drive offline")
        self.puts.append((local, remote))


def test_uploader_overwrites_best_in_vlsp_folder(tmp_path):
    drive = FakeDrive()
    uploader = DriveCheckpointUploader(drive, DRIVE_FRCRN_VLSP_DIR)
    checkpoint = tmp_path / "last_best_checkpoint.pt"
    uploader(checkpoint, 3, 12.5)
    assert uploader.remote_path == f"FA26_Capstone/Truong/FRCRN_vlsp/{BEST_FILENAME}"
    assert drive.made == [("FA26_Capstone/Truong/FRCRN_vlsp", True)]
    assert drive.puts == [(str(checkpoint), uploader.remote_path)]


def test_uploader_strips_slashes_from_remote_dir():
    uploader = DriveCheckpointUploader(FakeDrive(), "/a/b/")
    assert uploader.remote_path == "a/b/best.pt"


def test_uploader_swallows_drive_errors(tmp_path, caplog):
    uploader = DriveCheckpointUploader(FakeDrive(fail=True), DRIVE_FRCRN_VLSP_DIR, logging.getLogger("drive-test"))
    with caplog.at_level(logging.WARNING, logger="drive-test"):
        uploader(tmp_path / "x.pt", 1, 0.0)
    assert "upload failed" in caplog.text


def make_logger(tmp_path, name):
    log_file = tmp_path / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_file, encoding="utf-8"))
    return logger, log_file


def test_log_uploader_throttles_and_targets_vlsp_folder(tmp_path):
    drive = FakeDrive()
    logger, log_file = make_logger(tmp_path, "run-a")
    uploader = DriveLogUploader(drive, log_file, DRIVE_FRCRN_VLSP_DIR, interval=3600)
    logger.addHandler(uploader)
    logger.info("epoch 1")
    logger.info("epoch 2")
    assert uploader.remote_path == "FA26_Capstone/Truong/FRCRN_vlsp/run-a.log"
    assert drive.puts == [(str(log_file), uploader.remote_path)]
    assert "epoch 1" in log_file.read_text(encoding="utf-8")
    assert uploader.upload() is True
    assert len(drive.puts) == 2


def test_log_uploader_survives_drive_errors(tmp_path, capsys):
    logger, log_file = make_logger(tmp_path, "run-b")
    uploader = DriveLogUploader(FakeDrive(fail=True), log_file, DRIVE_FRCRN_VLSP_DIR, interval=0)
    logger.addHandler(uploader)
    logger.info("epoch 1")
    assert uploader.upload() is False
    assert "log upload failed" in capsys.readouterr().err


def test_log_uploader_flushes_trailing_lines_after_interval(tmp_path):
    drive = FakeDrive()
    logger, log_file = make_logger(tmp_path, "run-c")
    uploader = DriveLogUploader(drive, log_file, DRIVE_FRCRN_VLSP_DIR, interval=0.2)
    logger.addHandler(uploader)
    logger.info("first")
    logger.info("last line")
    assert len(drive.puts) == 1
    deadline = time.monotonic() + 3
    while len(drive.puts) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    uploader.close()
    assert len(drive.puts) == 2


def test_drive_file_backup_uploads_and_strips_paths(tmp_path, caplog):
    drive = FakeDrive()
    backup = DriveFileBackup(drive, "/kd-moe/stage2/", "resume.pt", logging.getLogger("drive-test"))
    local = tmp_path / "resume.pt"
    local.write_bytes(b"state")
    with caplog.at_level(logging.INFO, logger="drive-test"):
        backup(local)
    assert backup.remote_path == "kd-moe/stage2/resume.pt"
    assert drive.puts == [(str(local), "kd-moe/stage2/resume.pt")]
    assert "drive backup" in caplog.text


def test_drive_file_backup_swallows_errors(tmp_path, caplog):
    backup = DriveFileBackup(FakeDrive(fail=True), "kd-moe", "resume.pt", logging.getLogger("drive-test"))
    with caplog.at_level(logging.WARNING, logger="drive-test"):
        backup(tmp_path / "resume.pt")
    assert "drive backup failed" in caplog.text


def test_uploader_keep_history_adds_timestamped_copy(tmp_path):
    from scripts.drive_sync import utc_stamp

    drive = FakeDrive()
    uploader = DriveCheckpointUploader(drive, "a/b", keep_history=True)
    checkpoint = tmp_path / "last_best_checkpoint.pt"
    uploader(checkpoint, 7, -3.2)
    assert len(drive.puts) == 2
    assert drive.puts[0] == (str(checkpoint), "a/b/best.pt")
    history = drive.puts[1][1]
    assert history.startswith("a/b/best_e007_") and history.endswith("Z.pt")
    assert len(history) == len("a/b/best_e007_") + len(utc_stamp()) + len(".pt")


def test_uploader_without_history_keeps_single_upload(tmp_path):
    drive = FakeDrive()
    DriveCheckpointUploader(drive, "a/b")(tmp_path / "c.pt", 1, 0.0)
    assert len(drive.puts) == 1
