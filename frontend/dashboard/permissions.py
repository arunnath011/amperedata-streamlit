"""Dashboard permissions, sharing, and access control system.

This module provides comprehensive access control for dashboards including
user permissions, role-based access, sharing mechanisms, and security policies.
"""

import hashlib
import logging
import secrets
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from .exceptions import PermissionError
from .models import (
    DashboardConfig,
    DashboardPermission,
    DashboardRole,
    DashboardShare,
    PermissionLevel,
)
from .storage import DashboardStorage

logger = logging.getLogger(__name__)


class AccessResult(Enum):
    """Access check result enumeration."""

    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"
    NOT_FOUND = "not_found"
    INVALID_TOKEN = "invalid_token"


class DashboardPermissionManager:
    """Manages dashboard permissions and access control."""

    def __init__(self, storage: DashboardStorage):
        """Initialize permission manager.

        Args:
            storage: Dashboard storage instance
        """
        self.storage = storage
        self._permissions_cache: dict[str, list[DashboardPermission]] = {}
        self._role_hierarchy = {
            DashboardRole.VIEWER: [PermissionLevel.VIEW],
            DashboardRole.ANALYST: [PermissionLevel.VIEW, PermissionLevel.COMMENT],
            DashboardRole.RESEARCHER: [
                PermissionLevel.VIEW,
                PermissionLevel.COMMENT,
                PermissionLevel.EDIT,
            ],
            DashboardRole.ENGINEER: [
                PermissionLevel.VIEW,
                PermissionLevel.COMMENT,
                PermissionLevel.EDIT,
            ],
            DashboardRole.MANAGER: [
                PermissionLevel.VIEW,
                PermissionLevel.COMMENT,
                PermissionLevel.EDIT,
                PermissionLevel.ADMIN,
            ],
            DashboardRole.ADMIN: [
                PermissionLevel.VIEW,
                PermissionLevel.COMMENT,
                PermissionLevel.EDIT,
                PermissionLevel.ADMIN,
                PermissionLevel.OWNER,
            ],
        }

    async def grant_permission(
        self,
        dashboard_id: str,
        user_id: Optional[str] = None,
        role: Optional[DashboardRole] = None,
        permission_level: PermissionLevel = PermissionLevel.VIEW,
        granted_by: str = "system",
        expires_at: Optional[datetime] = None,
        conditions: Optional[dict[str, Any]] = None,
    ) -> DashboardPermission:
        """Grant permission to user or role.

        Args:
            dashboard_id: Dashboard ID
            user_id: User ID (for user-specific permissions)
            role: Role (for role-based permissions)
            permission_level: Permission level to grant
            granted_by: User who granted permission
            expires_at: Permission expiration time
            conditions: Additional permission conditions

        Returns:
            Created permission

        Raises:
            PermissionError: If permission cannot be granted
        """
        if not user_id and not role:
            raise PermissionError("Either user_id or role must be specified")

        if user_id and role:
            raise PermissionError("Cannot specify both user_id and role")

        # Check if dashboard exists
        dashboard = await self.storage.load_dashboard(dashboard_id)
        if not dashboard:
            raise PermissionError(f"Dashboard {dashboard_id} not found")

        # Check if granter has permission to grant
        if not await self.check_permission(dashboard_id, granted_by, PermissionLevel.ADMIN):
            raise PermissionError("Insufficient permissions to grant access")

        # Create permission
        permission = DashboardPermission(
            dashboard_id=dashboard_id,
            user_id=user_id,
            role=role,
            permission_level=permission_level,
            granted_by=granted_by,
            expires_at=expires_at,
            conditions=conditions or {},
        )

        # Save permission
        await self._save_permission(permission)

        # Update cache
        if dashboard_id in self._permissions_cache:
            self._permissions_cache[dashboard_id].append(permission)

        logger.info(f"Granted {permission_level} permission for dashboard {dashboard_id}")
        return permission

    async def revoke_permission(self, permission_id: str, revoked_by: str) -> bool:
        """Revoke permission.

        Args:
            permission_id: Permission ID to revoke
            revoked_by: User who revoked permission

        Returns:
            True if permission was revoked

        Raises:
            PermissionError: If permission cannot be revoked
        """
        permission = await self._load_permission(permission_id)
        if not permission:
            return False

        # Check if revoker has permission to revoke
        if not await self.check_permission(
            permission.dashboard_id, revoked_by, PermissionLevel.ADMIN
        ):
            raise PermissionError("Insufficient permissions to revoke access")

        # Remove permission
        success = await self._delete_permission(permission_id)

        if success:
            # Update cache
            if permission.dashboard_id in self._permissions_cache:
                self._permissions_cache[permission.dashboard_id] = [
                    p
                    for p in self._permissions_cache[permission.dashboard_id]
                    if p.id != permission_id
                ]

            logger.info(f"Revoked permission {permission_id}")

        return success

    async def check_permission(
        self,
        dashboard_id: str,
        user_id: str,
        required_level: PermissionLevel,
        user_roles: Optional[list[DashboardRole]] = None,
    ) -> bool:
        """Check if user has required permission level.

        Args:
            dashboard_id: Dashboard ID
            user_id: User ID
            required_level: Required permission level
            user_roles: User's roles (optional)

        Returns:
            True if user has required permission
        """
        try:
            # Check if user is dashboard owner
            dashboard = await self.storage.load_dashboard(dashboard_id)
            if dashboard and dashboard.created_by == user_id:
                return True

            # Get user permissions
            permissions = await self.get_user_permissions(dashboard_id, user_id, user_roles)

            # Check if any permission grants required level
            for permission in permissions:
                if self._is_permission_sufficient(permission.permission_level, required_level):
                    # Check if permission is still valid
                    if permission.expires_at and permission.expires_at < datetime.now():
                        continue

                    # Check additional conditions
                    if not self._check_permission_conditions(permission, user_id):
                        continue

                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking permission: {str(e)}")
            return False

    async def get_user_permissions(
        self,
        dashboard_id: str,
        user_id: str,
        user_roles: Optional[list[DashboardRole]] = None,
    ) -> list[DashboardPermission]:
        """Get all permissions for user on dashboard.

        Args:
            dashboard_id: Dashboard ID
            user_id: User ID
            user_roles: User's roles

        Returns:
            List of applicable permissions
        """
        all_permissions = await self.get_dashboard_permissions(dashboard_id)
        user_permissions = []

        for permission in all_permissions:
            # Direct user permission
            if permission.user_id == user_id:
                user_permissions.append(permission)

            # Role-based permission
            elif permission.role and user_roles and permission.role in user_roles:
                user_permissions.append(permission)

        return user_permissions

    async def get_dashboard_permissions(self, dashboard_id: str) -> list[DashboardPermission]:
        """Get all permissions for dashboard.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            List of dashboard permissions
        """
        if dashboard_id in self._permissions_cache:
            return self._permissions_cache[dashboard_id]

        try:
            permissions = await self._load_dashboard_permissions(dashboard_id)
            self._permissions_cache[dashboard_id] = permissions
            return permissions
        except Exception as e:
            logger.error(f"Failed to load permissions for dashboard {dashboard_id}: {str(e)}")
            return []

    async def get_user_dashboards(
        self,
        user_id: str,
        user_roles: Optional[list[DashboardRole]] = None,
        min_permission: PermissionLevel = PermissionLevel.VIEW,
    ) -> list[str]:
        """Get dashboards accessible to user.

        Args:
            user_id: User ID
            user_roles: User's roles
            min_permission: Minimum required permission level

        Returns:
            List of accessible dashboard IDs
        """
        accessible_dashboards = []

        try:
            # Get all dashboards (this could be optimized with proper indexing)
            all_dashboards = await self.storage.list_dashboards()

            for dashboard in all_dashboards:
                if await self.check_permission(dashboard.id, user_id, min_permission, user_roles):
                    accessible_dashboards.append(dashboard.id)

        except Exception as e:
            logger.error(f"Failed to get user dashboards: {str(e)}")

        return accessible_dashboards

    def _is_permission_sufficient(
        self, granted: PermissionLevel, required: PermissionLevel
    ) -> bool:
        """Check if granted permission level is sufficient for required level."""
        permission_hierarchy = [
            PermissionLevel.NONE,
            PermissionLevel.VIEW,
            PermissionLevel.COMMENT,
            PermissionLevel.EDIT,
            PermissionLevel.ADMIN,
            PermissionLevel.OWNER,
        ]

        try:
            granted_index = permission_hierarchy.index(granted)
            required_index = permission_hierarchy.index(required)
            return granted_index >= required_index
        except ValueError:
            return False

    def _check_permission_conditions(self, permission: DashboardPermission, user_id: str) -> bool:
        """Check additional permission conditions."""
        if not permission.conditions:
            return True

        # Example conditions (can be extended)
        conditions = permission.conditions

        # Time-based conditions
        if "time_restriction" in conditions:
            restriction = conditions["time_restriction"]
            current_hour = datetime.now().hour

            if "start_hour" in restriction and current_hour < restriction["start_hour"]:
                return False
            if "end_hour" in restriction and current_hour > restriction["end_hour"]:
                return False

        # IP-based conditions
        if "ip_whitelist" in conditions:
            # TODO: Implement IP checking (would need request context)
            pass

        return True

    async def _save_permission(self, permission: DashboardPermission) -> bool:
        """Save permission to storage."""
        # TODO: Implement permission storage
        return True

    async def _load_permission(self, permission_id: str) -> Optional[DashboardPermission]:
        """Load permission by ID."""
        # TODO: Implement permission loading
        return None

    async def _delete_permission(self, permission_id: str) -> bool:
        """Delete permission from storage."""
        # TODO: Implement permission deletion
        return True

    async def _load_dashboard_permissions(self, dashboard_id: str) -> list[DashboardPermission]:
        """Load all permissions for dashboard."""
        # TODO: Implement dashboard permissions loading
        return []


class ShareManager:
    """Manages dashboard sharing and public access."""

    def __init__(self, permission_manager: DashboardPermissionManager):
        """Initialize share manager.

        Args:
            permission_manager: Permission manager instance
        """
        self.permission_manager = permission_manager
        self._shares_cache: dict[str, DashboardShare] = {}

    async def create_share(
        self,
        dashboard_id: str,
        created_by: str,
        public: bool = False,
        password_protected: bool = False,
        password: Optional[str] = None,
        permission_level: PermissionLevel = PermissionLevel.VIEW,
        expires_at: Optional[datetime] = None,
        max_views: Optional[int] = None,
    ) -> DashboardShare:
        """Create dashboard share.

        Args:
            dashboard_id: Dashboard ID
            created_by: User creating share
            public: Make share public
            password_protected: Enable password protection
            password: Share password
            permission_level: Permission level for share
            expires_at: Share expiration time
            max_views: Maximum number of views

        Returns:
            Created share

        Raises:
            PermissionError: If share cannot be created
        """
        # Check if user can share dashboard
        if not await self.permission_manager.check_permission(
            dashboard_id, created_by, PermissionLevel.ADMIN
        ):
            raise PermissionError("Insufficient permissions to share dashboard")

        # Hash password if provided
        hashed_password = None
        if password_protected and password:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Create share
        share = DashboardShare(
            dashboard_id=dashboard_id,
            share_token=secrets.token_urlsafe(32),
            public=public,
            password_protected=password_protected,
            password=hashed_password,
            permission_level=permission_level,
            expires_at=expires_at,
            max_views=max_views,
            created_by=created_by,
        )

        # Save share
        await self._save_share(share)

        # Update cache
        self._shares_cache[share.share_token] = share

        logger.info(f"Created share {share.id} for dashboard {dashboard_id}")
        return share

    async def access_shared_dashboard(
        self,
        share_token: str,
        password: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> tuple[AccessResult, Optional[DashboardConfig]]:
        """Access dashboard via share token.

        Args:
            share_token: Share token
            password: Password (if required)
            user_id: User ID (for logging)

        Returns:
            Tuple of (access_result, dashboard_config)
        """
        try:
            # Load share
            share = await self._load_share_by_token(share_token)
            if not share:
                return AccessResult.NOT_FOUND, None

            # Check expiration
            if share.expires_at and share.expires_at < datetime.now():
                return AccessResult.EXPIRED, None

            # Check view limit
            if share.max_views and share.view_count >= share.max_views:
                return AccessResult.DENIED, None

            # Check password
            if share.password_protected:
                if not password:
                    return AccessResult.DENIED, None

                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                if hashed_password != share.password:
                    return AccessResult.INVALID_TOKEN, None

            # Load dashboard
            dashboard = await self.permission_manager.storage.load_dashboard(share.dashboard_id)
            if not dashboard:
                return AccessResult.NOT_FOUND, None

            # Increment view count
            share.view_count += 1
            await self._save_share(share)

            logger.info(f"Accessed shared dashboard {share.dashboard_id} via token {share_token}")
            return AccessResult.GRANTED, dashboard

        except Exception as e:
            logger.error(f"Error accessing shared dashboard: {str(e)}")
            return AccessResult.DENIED, None

    async def revoke_share(self, share_id: str, revoked_by: str) -> bool:
        """Revoke dashboard share.

        Args:
            share_id: Share ID
            revoked_by: User revoking share

        Returns:
            True if share was revoked
        """
        try:
            share = await self._load_share(share_id)
            if not share:
                return False

            # Check permissions
            if not await self.permission_manager.check_permission(
                share.dashboard_id, revoked_by, PermissionLevel.ADMIN
            ):
                raise PermissionError("Insufficient permissions to revoke share")

            # Delete share
            success = await self._delete_share(share_id)

            if success:
                # Update cache
                if share.share_token in self._shares_cache:
                    del self._shares_cache[share.share_token]

                logger.info(f"Revoked share {share_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to revoke share: {str(e)}")
            return False

    async def get_dashboard_shares(self, dashboard_id: str) -> list[DashboardShare]:
        """Get all shares for dashboard.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            List of dashboard shares
        """
        try:
            return await self._load_dashboard_shares(dashboard_id)
        except Exception as e:
            logger.error(f"Failed to get dashboard shares: {str(e)}")
            return []

    async def get_share_analytics(self, share_id: str) -> dict[str, Any]:
        """Get analytics for share.

        Args:
            share_id: Share ID

        Returns:
            Share analytics data
        """
        share = await self._load_share(share_id)
        if not share:
            return {}

        return {
            "share_id": share_id,
            "dashboard_id": share.dashboard_id,
            "view_count": share.view_count,
            "max_views": share.max_views,
            "created_at": share.created_at,
            "expires_at": share.expires_at,
            "is_expired": share.expires_at and share.expires_at < datetime.now(),
            "is_view_limited": share.max_views and share.view_count >= share.max_views,
            "remaining_views": max(0, (share.max_views or float("inf")) - share.view_count)
            if share.max_views
            else None,
        }

    async def _save_share(self, share: DashboardShare) -> bool:
        """Save share to storage."""
        # TODO: Implement share storage
        return True

    async def _load_share(self, share_id: str) -> Optional[DashboardShare]:
        """Load share by ID."""
        # TODO: Implement share loading
        return None

    async def _load_share_by_token(self, share_token: str) -> Optional[DashboardShare]:
        """Load share by token."""
        if share_token in self._shares_cache:
            return self._shares_cache[share_token]

        # TODO: Implement share loading by token
        return None

    async def _delete_share(self, share_id: str) -> bool:
        """Delete share from storage."""
        # TODO: Implement share deletion
        return True

    async def _load_dashboard_shares(self, dashboard_id: str) -> list[DashboardShare]:
        """Load all shares for dashboard."""
        # TODO: Implement dashboard shares loading
        return []


class AccessController:
    """Central access control coordinator."""

    def __init__(
        self,
        permission_manager: DashboardPermissionManager,
        share_manager: ShareManager,
    ):
        """Initialize access controller.

        Args:
            permission_manager: Permission manager instance
            share_manager: Share manager instance
        """
        self.permission_manager = permission_manager
        self.share_manager = share_manager
        self._access_log: list[dict[str, Any]] = []

    async def check_dashboard_access(
        self,
        dashboard_id: str,
        user_id: Optional[str] = None,
        share_token: Optional[str] = None,
        password: Optional[str] = None,
        required_permission: PermissionLevel = PermissionLevel.VIEW,
        user_roles: Optional[list[DashboardRole]] = None,
    ) -> tuple[AccessResult, Optional[DashboardConfig]]:
        """Check dashboard access via multiple methods.

        Args:
            dashboard_id: Dashboard ID
            user_id: User ID (for authenticated access)
            share_token: Share token (for shared access)
            password: Password (for protected shares)
            required_permission: Required permission level
            user_roles: User roles

        Returns:
            Tuple of (access_result, dashboard_config)
        """
        access_method = None
        result = AccessResult.DENIED
        dashboard = None

        try:
            # Try share token access first
            if share_token:
                access_method = "share_token"
                result, dashboard = await self.share_manager.access_shared_dashboard(
                    share_token, password, user_id
                )

                if result == AccessResult.GRANTED:
                    # Check if share permission level is sufficient
                    share = await self.share_manager._load_share_by_token(share_token)
                    if share and not self.permission_manager._is_permission_sufficient(
                        share.permission_level, required_permission
                    ):
                        result = AccessResult.DENIED
                        dashboard = None

            # Try authenticated user access
            elif user_id:
                access_method = "user_permission"
                if await self.permission_manager.check_permission(
                    dashboard_id, user_id, required_permission, user_roles
                ):
                    dashboard = await self.permission_manager.storage.load_dashboard(dashboard_id)
                    result = AccessResult.GRANTED if dashboard else AccessResult.NOT_FOUND
                else:
                    result = AccessResult.DENIED

            # Try public access
            else:
                access_method = "public"
                dashboard = await self.permission_manager.storage.load_dashboard(dashboard_id)
                if dashboard and dashboard.public:
                    result = AccessResult.GRANTED
                else:
                    result = AccessResult.DENIED

            # Log access attempt
            self._log_access_attempt(
                dashboard_id=dashboard_id,
                user_id=user_id,
                share_token=share_token,
                access_method=access_method,
                result=result,
                required_permission=required_permission,
            )

            return result, dashboard

        except Exception as e:
            logger.error(f"Error checking dashboard access: {str(e)}")
            self._log_access_attempt(
                dashboard_id=dashboard_id,
                user_id=user_id,
                share_token=share_token,
                access_method=access_method,
                result=AccessResult.DENIED,
                error=str(e),
            )
            return AccessResult.DENIED, None

    def _log_access_attempt(
        self,
        dashboard_id: str,
        user_id: Optional[str] = None,
        share_token: Optional[str] = None,
        access_method: Optional[str] = None,
        result: AccessResult = AccessResult.DENIED,
        required_permission: Optional[PermissionLevel] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log access attempt for auditing."""
        log_entry = {
            "timestamp": datetime.now(),
            "dashboard_id": dashboard_id,
            "user_id": user_id,
            "share_token": share_token[:8] + "..."
            if share_token
            else None,  # Partial token for security
            "access_method": access_method,
            "result": result.value,
            "required_permission": required_permission.value if required_permission else None,
            "error": error,
        }

        self._access_log.append(log_entry)

        # Keep only recent entries (last 1000)
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]

        # Log to system logger
        if result == AccessResult.GRANTED:
            logger.info(f"Dashboard access granted: {dashboard_id} via {access_method}")
        else:
            logger.warning(
                f"Dashboard access denied: {dashboard_id} via {access_method} - {result.value}"
            )

    def get_access_log(
        self, dashboard_id: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get access log entries.

        Args:
            dashboard_id: Filter by dashboard ID
            limit: Maximum number of entries

        Returns:
            List of access log entries
        """
        filtered_log = self._access_log

        if dashboard_id:
            filtered_log = [
                entry for entry in filtered_log if entry["dashboard_id"] == dashboard_id
            ]

        # Return most recent entries
        return filtered_log[-limit:]
