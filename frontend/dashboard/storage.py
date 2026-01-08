"""Dashboard storage, versioning, and snapshot management.

This module provides persistent storage for dashboards with version control,
snapshot capabilities, and database integration using SQLAlchemy.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

try:
    from sqlalchemy import Boolean, Column, DateTime, Integer, LargeBinary, String, Text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from .exceptions import ExportError, StorageError, VersionError
from .models import (
    DashboardConfig,
    DashboardSnapshot,
    DashboardVersion,
    ExportFormat,
    ScheduleConfig,
)

logger = logging.getLogger(__name__)

# SQLAlchemy models (if available)
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    class DashboardRecord(Base):
        """SQLAlchemy model for dashboard storage."""

        __tablename__ = "dashboards"

        id = Column(String, primary_key=True)
        name = Column(String, nullable=False)
        description = Column(Text)
        config_json = Column(Text, nullable=False)
        created_by = Column(String, nullable=False)
        created_at = Column(DateTime, default=datetime.now)
        updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
        version = Column(Integer, default=1)
        is_public = Column(Boolean, default=False)
        is_deleted = Column(Boolean, default=False)

    class DashboardVersionRecord(Base):
        """SQLAlchemy model for dashboard versions."""

        __tablename__ = "dashboard_versions"

        id = Column(String, primary_key=True)
        dashboard_id = Column(String, nullable=False)
        version_number = Column(Integer, nullable=False)
        config_json = Column(Text, nullable=False)
        change_summary = Column(Text)
        created_by = Column(String, nullable=False)
        created_at = Column(DateTime, default=datetime.now)
        is_published = Column(Boolean, default=False)
        checksum = Column(String)

    class DashboardSnapshotRecord(Base):
        """SQLAlchemy model for dashboard snapshots."""

        __tablename__ = "dashboard_snapshots"

        id = Column(String, primary_key=True)
        dashboard_id = Column(String, nullable=False)
        name = Column(String, nullable=False)
        description = Column(Text)
        format = Column(String, nullable=False)
        file_path = Column(String)
        file_size = Column(Integer)
        file_data = Column(LargeBinary)
        created_by = Column(String, nullable=False)
        created_at = Column(DateTime, default=datetime.now)
        scheduled = Column(Boolean, default=False)
        schedule_config_json = Column(Text)


class DashboardStorage:
    """Handles dashboard persistence and retrieval."""

    def __init__(self, storage_path: Optional[Path] = None, database_url: Optional[str] = None):
        """Initialize dashboard storage.

        Args:
            storage_path: File system storage path (fallback)
            database_url: Database connection URL
        """
        self.storage_path = storage_path or Path("./dashboards")
        self.database_url = database_url
        self._engine = None
        self._session_maker = None

        # Ensure storage directory exists
        if not SQLALCHEMY_AVAILABLE:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize database if available
        if SQLALCHEMY_AVAILABLE and database_url:
            self._init_database()

    def _init_database(self) -> None:
        """Initialize database connection."""
        try:
            self._engine = create_async_engine(self.database_url)
            self._session_maker = sessionmaker(
                bind=self._engine, class_=AsyncSession, expire_on_commit=False
            )
            logger.info("Database storage initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            self._engine = None
            self._session_maker = None

    async def save_dashboard(self, config: DashboardConfig) -> bool:
        """Save dashboard configuration.

        Args:
            config: Dashboard configuration

        Returns:
            True if saved successfully

        Raises:
            StorageError: If save operation fails
        """
        try:
            if self._session_maker:
                return await self._save_to_database(config)
            else:
                return self._save_to_file(config)
        except Exception as e:
            logger.error(f"Failed to save dashboard {config.id}: {str(e)}")
            raise StorageError(f"Save operation failed: {str(e)}")

    async def load_dashboard(self, dashboard_id: str) -> Optional[DashboardConfig]:
        """Load dashboard configuration.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            Dashboard configuration or None if not found

        Raises:
            StorageError: If load operation fails
        """
        try:
            if self._session_maker:
                return await self._load_from_database(dashboard_id)
            else:
                return self._load_from_file(dashboard_id)
        except Exception as e:
            logger.error(f"Failed to load dashboard {dashboard_id}: {str(e)}")
            raise StorageError(f"Load operation failed: {str(e)}")

    async def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete dashboard (soft delete).

        Args:
            dashboard_id: Dashboard ID

        Returns:
            True if deleted successfully
        """
        try:
            if self._session_maker:
                return await self._delete_from_database(dashboard_id)
            else:
                return self._delete_from_file(dashboard_id)
        except Exception as e:
            logger.error(f"Failed to delete dashboard {dashboard_id}: {str(e)}")
            return False

    async def list_dashboards(
        self,
        user_id: Optional[str] = None,
        public_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DashboardConfig]:
        """List dashboards with filtering.

        Args:
            user_id: Filter by user ID
            public_only: Show only public dashboards
            limit: Maximum number of results
            offset: Result offset for pagination

        Returns:
            List of dashboard configurations
        """
        try:
            if self._session_maker:
                return await self._list_from_database(user_id, public_only, limit, offset)
            else:
                return self._list_from_files(user_id, public_only, limit, offset)
        except Exception as e:
            logger.error(f"Failed to list dashboards: {str(e)}")
            return []

    async def _save_to_database(self, config: DashboardConfig) -> bool:
        """Save dashboard to database."""
        if not SQLALCHEMY_AVAILABLE or not self._session_maker:
            return False

        async with self._session_maker() as session:
            try:
                # Check if dashboard exists
                existing = await session.get(DashboardRecord, config.id)

                if existing:
                    # Update existing
                    existing.name = config.name
                    existing.description = config.description
                    existing.config_json = config.json()
                    existing.updated_at = datetime.now()
                    existing.version = config.version
                    existing.is_public = config.public
                else:
                    # Create new
                    record = DashboardRecord(
                        id=config.id,
                        name=config.name,
                        description=config.description,
                        config_json=config.json(),
                        created_by=config.created_by,
                        version=config.version,
                        is_public=config.public,
                    )
                    session.add(record)

                await session.commit()
                return True

            except Exception as e:
                await session.rollback()
                raise e

    async def _load_from_database(self, dashboard_id: str) -> Optional[DashboardConfig]:
        """Load dashboard from database."""
        if not SQLALCHEMY_AVAILABLE or not self._session_maker:
            return None

        async with self._session_maker() as session:
            record = await session.get(DashboardRecord, dashboard_id)
            if record and not record.is_deleted:
                return DashboardConfig.parse_raw(record.config_json)
            return None

    async def _delete_from_database(self, dashboard_id: str) -> bool:
        """Soft delete dashboard from database."""
        if not SQLALCHEMY_AVAILABLE or not self._session_maker:
            return False

        async with self._session_maker() as session:
            try:
                record = await session.get(DashboardRecord, dashboard_id)
                if record:
                    record.is_deleted = True
                    record.updated_at = datetime.now()
                    await session.commit()
                    return True
                return False
            except Exception as e:
                await session.rollback()
                raise e

    async def _list_from_database(
        self, user_id: Optional[str], public_only: bool, limit: int, offset: int
    ) -> list[DashboardConfig]:
        """List dashboards from database."""
        if not SQLALCHEMY_AVAILABLE or not self._session_maker:
            return []

        # TODO: Implement proper SQLAlchemy query with filters
        # For now, return empty list
        return []

    def _save_to_file(self, config: DashboardConfig) -> bool:
        """Save dashboard to file system."""
        file_path = self.storage_path / f"{config.id}.json"

        try:
            with open(file_path, "w") as f:
                f.write(config.json(indent=2))

            logger.info(f"Saved dashboard {config.id} to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save dashboard to file: {str(e)}")
            return False

    def _load_from_file(self, dashboard_id: str) -> Optional[DashboardConfig]:
        """Load dashboard from file system."""
        file_path = self.storage_path / f"{dashboard_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                return DashboardConfig.parse_raw(f.read())
        except Exception as e:
            logger.error(f"Failed to load dashboard from file: {str(e)}")
            return None

    def _delete_from_file(self, dashboard_id: str) -> bool:
        """Delete dashboard file."""
        file_path = self.storage_path / f"{dashboard_id}.json"

        try:
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete dashboard file: {str(e)}")
            return False

    def _list_from_files(
        self, user_id: Optional[str], public_only: bool, limit: int, offset: int
    ) -> list[DashboardConfig]:
        """List dashboards from file system."""
        dashboards = []

        try:
            json_files = list(self.storage_path.glob("*.json"))

            for file_path in json_files[offset : offset + limit]:
                try:
                    config = self._load_from_file(file_path.stem)
                    if config:
                        # Apply filters
                        if user_id and config.created_by != user_id:
                            continue
                        if public_only and not config.public:
                            continue

                        dashboards.append(config)
                except Exception as e:
                    logger.error(f"Failed to load dashboard from {file_path}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Failed to list dashboard files: {str(e)}")

        return dashboards


class VersionManager:
    """Manages dashboard versioning and change tracking."""

    def __init__(self, storage: DashboardStorage):
        """Initialize version manager.

        Args:
            storage: Dashboard storage instance
        """
        self.storage = storage
        self._versions_cache: dict[str, list[DashboardVersion]] = {}

    async def create_version(
        self,
        dashboard_id: str,
        config: DashboardConfig,
        change_summary: Optional[str] = None,
        created_by: str = "system",
    ) -> DashboardVersion:
        """Create new dashboard version.

        Args:
            dashboard_id: Dashboard ID
            config: Dashboard configuration
            change_summary: Summary of changes
            created_by: User who created version

        Returns:
            Created version

        Raises:
            VersionError: If version creation fails
        """
        try:
            # Get current versions
            versions = await self.get_versions(dashboard_id)
            next_version = len(versions) + 1

            # Calculate checksum
            config_json = config.json()
            checksum = hashlib.sha256(config_json.encode()).hexdigest()

            # Check for duplicate versions
            for version in versions:
                if version.config.json() == config_json:
                    raise VersionError(
                        f"Identical version already exists: {version.version_number}"
                    )

            # Create version
            version = DashboardVersion(
                dashboard_id=dashboard_id,
                version_number=next_version,
                config=config,
                change_summary=change_summary,
                created_by=created_by,
            )

            # Save version
            await self._save_version(version, checksum)

            # Update cache
            if dashboard_id in self._versions_cache:
                self._versions_cache[dashboard_id].append(version)

            logger.info(f"Created version {next_version} for dashboard {dashboard_id}")
            return version

        except Exception as e:
            logger.error(f"Failed to create version: {str(e)}")
            raise VersionError(f"Version creation failed: {str(e)}")

    async def get_versions(self, dashboard_id: str) -> list[DashboardVersion]:
        """Get all versions for dashboard.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            List of versions sorted by version number
        """
        if dashboard_id in self._versions_cache:
            return self._versions_cache[dashboard_id]

        try:
            versions = await self._load_versions(dashboard_id)
            versions.sort(key=lambda v: v.version_number)
            self._versions_cache[dashboard_id] = versions
            return versions

        except Exception as e:
            logger.error(f"Failed to get versions for dashboard {dashboard_id}: {str(e)}")
            return []

    async def get_version(self, dashboard_id: str, version_number: int) -> Optional[DashboardVersion]:
        """Get specific version.

        Args:
            dashboard_id: Dashboard ID
            version_number: Version number

        Returns:
            Version or None if not found
        """
        versions = await self.get_versions(dashboard_id)
        for version in versions:
            if version.version_number == version_number:
                return version
        return None

    async def publish_version(self, dashboard_id: str, version_number: int) -> bool:
        """Publish specific version.

        Args:
            dashboard_id: Dashboard ID
            version_number: Version number to publish

        Returns:
            True if published successfully
        """
        version = await self.get_version(dashboard_id, version_number)
        if not version:
            return False

        try:
            # Unpublish other versions
            versions = await self.get_versions(dashboard_id)
            for v in versions:
                if v.is_published:
                    v.is_published = False
                    await self._save_version(v)

            # Publish target version
            version.is_published = True
            await self._save_version(version)

            logger.info(f"Published version {version_number} for dashboard {dashboard_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish version: {str(e)}")
            return False

    async def compare_versions(
        self, dashboard_id: str, version1: int, version2: int
    ) -> dict[str, Any]:
        """Compare two versions.

        Args:
            dashboard_id: Dashboard ID
            version1: First version number
            version2: Second version number

        Returns:
            Comparison result
        """
        v1 = await self.get_version(dashboard_id, version1)
        v2 = await self.get_version(dashboard_id, version2)

        if not v1 or not v2:
            return {"error": "One or both versions not found"}

        # Simple comparison - can be enhanced
        config1 = v1.config.dict()
        config2 = v2.config.dict()

        differences = {}

        # Compare basic fields
        for field in ["name", "description", "theme"]:
            if config1.get(field) != config2.get(field):
                differences[field] = {
                    f"version_{version1}": config1.get(field),
                    f"version_{version2}": config2.get(field),
                }

        # Compare widgets
        widgets1 = {w["id"]: w for w in config1.get("widgets", [])}
        widgets2 = {w["id"]: w for w in config2.get("widgets", [])}

        added_widgets = set(widgets2.keys()) - set(widgets1.keys())
        removed_widgets = set(widgets1.keys()) - set(widgets2.keys())
        modified_widgets = []

        for widget_id in set(widgets1.keys()) & set(widgets2.keys()):
            if widgets1[widget_id] != widgets2[widget_id]:
                modified_widgets.append(widget_id)

        differences["widgets"] = {
            "added": list(added_widgets),
            "removed": list(removed_widgets),
            "modified": modified_widgets,
        }

        return {
            "version1": version1,
            "version2": version2,
            "differences": differences,
            "summary": {
                "total_changes": len(differences),
                "widgets_changed": len(added_widgets)
                + len(removed_widgets)
                + len(modified_widgets),
            },
        }

    async def rollback_to_version(self, dashboard_id: str, version_number: int) -> bool:
        """Rollback dashboard to specific version.

        Args:
            dashboard_id: Dashboard ID
            version_number: Version to rollback to

        Returns:
            True if rollback successful
        """
        version = await self.get_version(dashboard_id, version_number)
        if not version:
            return False

        try:
            # Save current state as new version first
            current_config = await self.storage.load_dashboard(dashboard_id)
            if current_config:
                await self.create_version(
                    dashboard_id,
                    current_config,
                    f"Auto-save before rollback to version {version_number}",
                    "system",
                )

            # Update dashboard with rollback version
            rollback_config = version.config.copy()
            rollback_config.updated_at = datetime.now()
            rollback_config.version += 1

            success = await self.storage.save_dashboard(rollback_config)

            if success:
                logger.info(f"Rolled back dashboard {dashboard_id} to version {version_number}")

            return success

        except Exception as e:
            logger.error(f"Failed to rollback dashboard: {str(e)}")
            return False

    async def _save_version(self, version: DashboardVersion, checksum: Optional[str] = None) -> bool:
        """Save version to storage."""
        # TODO: Implement version storage (database or file system)
        # For now, just log the operation
        logger.info(f"Saving version {version.version_number} for dashboard {version.dashboard_id}")
        return True

    async def _load_versions(self, dashboard_id: str) -> list[DashboardVersion]:
        """Load versions from storage."""
        # TODO: Implement version loading
        # For now, return empty list
        return []


class SnapshotManager:
    """Manages dashboard snapshots and scheduled exports."""

    def __init__(self, storage: DashboardStorage, export_path: Optional[Path] = None):
        """Initialize snapshot manager.

        Args:
            storage: Dashboard storage instance
            export_path: Path for snapshot files
        """
        self.storage = storage
        self.export_path = export_path or Path("./snapshots")
        self.export_path.mkdir(parents=True, exist_ok=True)

    async def create_snapshot(
        self,
        dashboard_id: str,
        name: str,
        format: ExportFormat,
        description: Optional[str] = None,
        created_by: str = "system",
    ) -> DashboardSnapshot:
        """Create dashboard snapshot.

        Args:
            dashboard_id: Dashboard ID
            name: Snapshot name
            format: Export format
            description: Snapshot description
            created_by: User who created snapshot

        Returns:
            Created snapshot

        Raises:
            ExportError: If snapshot creation fails
        """
        try:
            # Load dashboard
            config = await self.storage.load_dashboard(dashboard_id)
            if not config:
                raise ExportError(f"Dashboard {dashboard_id} not found")

            # Create snapshot
            snapshot = DashboardSnapshot(
                dashboard_id=dashboard_id,
                name=name,
                description=description,
                format=format,
                created_by=created_by,
            )

            # Export dashboard
            file_path, file_size = await self._export_dashboard(config, snapshot, format)

            snapshot.file_path = str(file_path)
            snapshot.file_size = file_size

            # Save snapshot metadata
            await self._save_snapshot(snapshot)

            logger.info(f"Created snapshot {snapshot.id} for dashboard {dashboard_id}")
            return snapshot

        except Exception as e:
            logger.error(f"Failed to create snapshot: {str(e)}")
            raise ExportError(f"Snapshot creation failed: {str(e)}")

    async def schedule_snapshots(self, dashboard_id: str, schedule_config: ScheduleConfig) -> bool:
        """Schedule automatic snapshots.

        Args:
            dashboard_id: Dashboard ID
            schedule_config: Schedule configuration

        Returns:
            True if scheduling successful
        """
        try:
            # TODO: Implement scheduling using Celery or similar
            # For now, just log the operation
            logger.info(f"Scheduled snapshots for dashboard {dashboard_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to schedule snapshots: {str(e)}")
            return False

    async def get_snapshots(self, dashboard_id: str) -> list[DashboardSnapshot]:
        """Get all snapshots for dashboard.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            List of snapshots
        """
        try:
            return await self._load_snapshots(dashboard_id)
        except Exception as e:
            logger.error(f"Failed to get snapshots: {str(e)}")
            return []

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete snapshot.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            True if deleted successfully
        """
        try:
            # TODO: Implement snapshot deletion
            logger.info(f"Deleted snapshot {snapshot_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete snapshot: {str(e)}")
            return False

    async def _export_dashboard(
        self, config: DashboardConfig, snapshot: DashboardSnapshot, format: ExportFormat
    ) -> tuple[Path, int]:
        """Export dashboard to file.

        Args:
            config: Dashboard configuration
            snapshot: Snapshot metadata
            format: Export format

        Returns:
            Tuple of (file_path, file_size)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.name}_{timestamp}.{format.value}"
        file_path = self.export_path / filename

        if format == ExportFormat.JSON:
            # Export as JSON
            with open(file_path, "w") as f:
                f.write(config.json(indent=2))

        elif format == ExportFormat.PDF:
            # TODO: Implement PDF export
            # For now, create placeholder file
            with open(file_path, "w") as f:
                f.write(f"PDF export placeholder for dashboard: {config.name}")

        elif format == ExportFormat.PNG:
            # TODO: Implement PNG export (screenshot)
            # For now, create placeholder file
            with open(file_path, "w") as f:
                f.write(f"PNG export placeholder for dashboard: {config.name}")

        else:
            raise ExportError(f"Unsupported export format: {format}")

        file_size = file_path.stat().st_size
        return file_path, file_size

    async def _save_snapshot(self, snapshot: DashboardSnapshot) -> bool:
        """Save snapshot metadata."""
        # TODO: Implement snapshot metadata storage
        return True

    async def _load_snapshots(self, dashboard_id: str) -> list[DashboardSnapshot]:
        """Load snapshots from storage."""
        # TODO: Implement snapshot loading
        return []
