"""Verified Drive checkpoint transfers using Google's resumable media API.

Use gdrive-fsspec for cached authentication/folders only. Its buffered writer
creates duplicate names and may leave large uploads incomplete.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path


class DriveCheckpointStore:
    def __init__(self):
        from gdrive_fsspec import GoogleDriveFileSystem
        self.fs = GoogleDriveFileSystem(
            use_listings_cache=True, skip_instance_cache=True, token="cache")
        self.files = self.fs.files
        self.folder_ids = {}

    def makedirs(self, path, exist_ok=True):
        self.fs.makedirs(path, exist_ok=exist_ok)
        self.folder_ids[path] = self.fs.info(path)["id"]

    def _folder(self, path):
        if path not in self.folder_ids:
            self.folder_ids[path] = self.fs.info(path)["id"]
        return self.folder_ids[path]

    @staticmethod
    def _escape(value):
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _find(self, path):
        parent, name = path.rsplit("/", 1)
        query = (f"'{self._escape(self._folder(parent))}' in parents and "
                 f"name = '{self._escape(name)}' and trashed = false")
        response = self.files.list(
            q=query, fields="files(id,name,size,md5Checksum,modifiedTime)",
            orderBy="modifiedTime desc", pageSize=100,
            supportsAllDrives=True, includeItemsFromAllDrives=True).execute(num_retries=3)
        entries = response.get("files", [])
        return entries[0] if entries else None

    def exists(self, path):
        return self._find(path) is not None

    def invalidate_cache(self, path):
        self.fs.invalidate_cache(path)

    def ls(self, path):
        return self.files.list(
            q=f"'{self._escape(self._folder(path))}' in parents and trashed = false",
            fields="files(id,name,size)", supportsAllDrives=True,
            includeItemsFromAllDrives=True).execute(num_retries=3).get("files", [])

    def put_file(self, local, remote):
        from googleapiclient.http import MediaFileUpload
        source = Path(local)
        digest = hashlib.md5()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        existing = self._find(remote)
        media = MediaFileUpload(str(source), chunksize=8 * 1024 * 1024, resumable=True)
        options = dict(media_body=media, fields="id,size,md5Checksum", supportsAllDrives=True)
        if existing:
            request = self.files.update(fileId=existing["id"], **options)
        else:
            parent, name = remote.rsplit("/", 1)
            request = self.files.create(body={"name": name, "parents": [self._folder(parent)]}, **options)
        result = None
        while result is None:
            _, result = request.next_chunk(num_retries=3)
        if int(result.get("size", -1)) != source.stat().st_size or result.get("md5Checksum") != digest.hexdigest():
            raise IOError("Drive upload size/checksum mismatch")

    def get_file(self, remote, local):
        from googleapiclient.http import MediaIoBaseDownload
        entry = self._find(remote)
        if entry is None:
            raise FileNotFoundError(remote)
        with Path(local).open("wb") as stream:
            download = MediaIoBaseDownload(
                stream, self.files.get_media(fileId=entry["id"], supportsAllDrives=True),
                chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = download.next_chunk(num_retries=3)

    def open(self, remote, mode):
        if mode != "rb":
            raise ValueError("Only small backup metadata reads are supported")
        entry = self._find(remote)
        if entry is None:
            raise FileNotFoundError(remote)
        if int(entry.get("size", 0)) > 1024 * 1024:
            raise ValueError("Use get_file for large checkpoint files")
        data = self.files.get_media(fileId=entry["id"], supportsAllDrives=True).execute(num_retries=3)
        return io.BytesIO(data)

    def rm(self, remote):
        # Called only for the previous versioned pair recorded by our manifest.
        entry = self._find(remote)
        if entry:
            self.files.delete(fileId=entry["id"], supportsAllDrives=True).execute(num_retries=3)
