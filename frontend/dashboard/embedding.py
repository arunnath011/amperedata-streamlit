"""Dashboard embedding and integration system.

This module provides capabilities for embedding dashboards in external applications,
generating iframe code, API endpoints, and managing embed configurations.
"""

import base64
import logging
from datetime import datetime, timedelta
from typing import Optional, Any
from urllib.parse import urlencode

from .exceptions import EmbedError
from .models import EmbedConfig, EmbedType
from .permissions import AccessController, AccessResult

logger = logging.getLogger(__name__)


class EmbedManager:
    """Manages dashboard embedding configurations and access."""

    def __init__(
        self,
        access_controller: AccessController,
        base_url: str = "http://localhost:8000",
    ):
        """Initialize embed manager.

        Args:
            access_controller: Access controller instance
            base_url: Base URL for embed links
        """
        self.access_controller = access_controller
        self.base_url = base_url.rstrip("/")
        self._embed_configs: dict[str, EmbedConfig] = {}

    async def create_embed(
        self,
        dashboard_id: str,
        embed_type: EmbedType,
        created_by: str,
        public: bool = False,
        password_protected: bool = False,
        password: Optional[str] = None,
        allowed_domains: Optional[list[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        auto_refresh: bool = True,
        show_toolbar: bool = False,
        show_filters: bool = True,
        expires_at: Optional[datetime] = None,
    ) -> EmbedConfig:
        """Create dashboard embed configuration.

        Args:
            dashboard_id: Dashboard ID
            embed_type: Type of embed
            created_by: User creating embed
            public: Make embed public
            password_protected: Enable password protection
            password: Embed password
            allowed_domains: Allowed embedding domains
            width: Embed width
            height: Embed height
            auto_refresh: Enable auto-refresh
            show_toolbar: Show dashboard toolbar
            show_filters: Show filter widgets
            expires_at: Embed expiration time

        Returns:
            Created embed configuration

        Raises:
            EmbedError: If embed creation fails
        """
        try:
            # Check if user can create embed for dashboard
            result, dashboard = await self.access_controller.check_dashboard_access(
                dashboard_id=dashboard_id,
                user_id=created_by,
                required_permission="admin",
            )

            if result != AccessResult.GRANTED:
                raise EmbedError("Insufficient permissions to create embed")

            # Create embed configuration
            embed_config = EmbedConfig(
                dashboard_id=dashboard_id,
                embed_type=embed_type,
                public=public,
                password_protected=password_protected,
                password=password,
                allowed_domains=allowed_domains or [],
                width=width,
                height=height,
                auto_refresh=auto_refresh,
                show_toolbar=show_toolbar,
                show_filters=show_filters,
                expires_at=expires_at,
                created_by=created_by,
            )

            # Save embed configuration
            await self._save_embed_config(embed_config)

            # Cache configuration
            self._embed_configs[embed_config.id] = embed_config

            logger.info(f"Created embed {embed_config.id} for dashboard {dashboard_id}")
            return embed_config

        except Exception as e:
            logger.error(f"Failed to create embed: {str(e)}")
            raise EmbedError(f"Embed creation failed: {str(e)}")

    async def get_embed_config(self, embed_id: str) -> Optional[EmbedConfig]:
        """Get embed configuration by ID.

        Args:
            embed_id: Embed configuration ID

        Returns:
            Embed configuration or None
        """
        if embed_id in self._embed_configs:
            return self._embed_configs[embed_id]

        try:
            config = await self._load_embed_config(embed_id)
            if config:
                self._embed_configs[embed_id] = config
            return config
        except Exception as e:
            logger.error(f"Failed to load embed config: {str(e)}")
            return None

    async def validate_embed_access(
        self,
        embed_id: str,
        domain: Optional[str] = None,
        password: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Validate embed access.

        Args:
            embed_id: Embed configuration ID
            domain: Requesting domain
            password: Provided password

        Returns:
            Tuple of (is_valid, error_message)
        """
        config = await self.get_embed_config(embed_id)
        if not config:
            return False, "Embed configuration not found"

        # Check expiration
        if config.expires_at and config.expires_at < datetime.now():
            return False, "Embed has expired"

        # Check domain restrictions
        if config.allowed_domains and domain:
            if not any(self._domain_matches(domain, allowed) for allowed in config.allowed_domains):
                return False, f"Domain {domain} not allowed"

        # Check password
        if config.password_protected:
            if not password or password != config.password:
                return False, "Invalid password"

        return True, None

    def _domain_matches(self, domain: str, pattern: str) -> bool:
        """Check if domain matches pattern (supports wildcards).

        Args:
            domain: Domain to check
            pattern: Pattern to match against

        Returns:
            True if domain matches pattern
        """
        if pattern == "*":
            return True

        if pattern.startswith("*."):
            # Subdomain wildcard
            suffix = pattern[2:]
            return domain == suffix or domain.endswith("." + suffix)

        return domain == pattern

    async def revoke_embed(self, embed_id: str, revoked_by: str) -> bool:
        """Revoke embed configuration.

        Args:
            embed_id: Embed configuration ID
            revoked_by: User revoking embed

        Returns:
            True if embed was revoked
        """
        config = await self.get_embed_config(embed_id)
        if not config:
            return False

        try:
            # Check permissions
            result, _ = await self.access_controller.check_dashboard_access(
                dashboard_id=config.dashboard_id,
                user_id=revoked_by,
                required_permission="admin",
            )

            if result != AccessResult.GRANTED:
                raise EmbedError("Insufficient permissions to revoke embed")

            # Delete embed configuration
            success = await self._delete_embed_config(embed_id)

            if success:
                # Remove from cache
                if embed_id in self._embed_configs:
                    del self._embed_configs[embed_id]

                logger.info(f"Revoked embed {embed_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to revoke embed: {str(e)}")
            return False

    async def list_dashboard_embeds(self, dashboard_id: str) -> list[EmbedConfig]:
        """List all embeds for dashboard.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            List of embed configurations
        """
        try:
            return await self._load_dashboard_embeds(dashboard_id)
        except Exception as e:
            logger.error(f"Failed to list dashboard embeds: {str(e)}")
            return []

    async def _save_embed_config(self, config: EmbedConfig) -> bool:
        """Save embed configuration to storage."""
        # TODO: Implement embed config storage
        return True

    async def _load_embed_config(self, embed_id: str) -> Optional[EmbedConfig]:
        """Load embed configuration from storage."""
        # TODO: Implement embed config loading
        return None

    async def _delete_embed_config(self, embed_id: str) -> bool:
        """Delete embed configuration from storage."""
        # TODO: Implement embed config deletion
        return True

    async def _load_dashboard_embeds(self, dashboard_id: str) -> list[EmbedConfig]:
        """Load all embed configurations for dashboard."""
        # TODO: Implement dashboard embeds loading
        return []


class IFrameGenerator:
    """Generates iframe embed code for dashboards."""

    def __init__(self, embed_manager: EmbedManager):
        """Initialize iframe generator.

        Args:
            embed_manager: Embed manager instance
        """
        self.embed_manager = embed_manager

    def generate_iframe_code(
        self,
        embed_config: EmbedConfig,
        custom_width: Optional[int] = None,
        custom_height: Optional[int] = None,
        additional_params: Optional[dict[str, str]] = None,
    ) -> str:
        """Generate iframe HTML code for dashboard embed.

        Args:
            embed_config: Embed configuration
            custom_width: Custom width override
            custom_height: Custom height override
            additional_params: Additional URL parameters

        Returns:
            HTML iframe code
        """
        # Build embed URL
        embed_url = self._build_embed_url(embed_config, additional_params)

        # Determine dimensions
        width = custom_width or embed_config.width or 800
        height = custom_height or embed_config.height or 600

        # Generate iframe attributes
        iframe_attrs = {
            "src": embed_url,
            "width": str(width),
            "height": str(height),
            "frameborder": "0",
            "scrolling": "no",
            "allowfullscreen": "true",
        }

        # Add security attributes
        if embed_config.allowed_domains:
            # Add sandbox restrictions for security
            iframe_attrs["sandbox"] = "allow-scripts allow-same-origin allow-forms"

        # Build iframe HTML
        attrs_str = " ".join(f'{key}="{value}"' for key, value in iframe_attrs.items())
        iframe_html = f"<iframe {attrs_str}></iframe>"

        return iframe_html

    def generate_responsive_iframe_code(
        self,
        embed_config: EmbedConfig,
        aspect_ratio: str = "16:9",
        additional_params: Optional[dict[str, str]] = None,
    ) -> str:
        """Generate responsive iframe HTML code.

        Args:
            embed_config: Embed configuration
            aspect_ratio: Aspect ratio (e.g., "16:9", "4:3")
            additional_params: Additional URL parameters

        Returns:
            HTML code with responsive wrapper
        """
        # Calculate padding-bottom for aspect ratio
        ratio_parts = aspect_ratio.split(":")
        if len(ratio_parts) == 2:
            width_ratio = float(ratio_parts[0])
            height_ratio = float(ratio_parts[1])
            padding_bottom = (height_ratio / width_ratio) * 100
        else:
            padding_bottom = 56.25  # Default 16:9

        # Build embed URL
        embed_url = self._build_embed_url(embed_config, additional_params)

        # Generate responsive HTML
        responsive_html = f"""
<div style="position: relative; width: 100%; height: 0; padding-bottom: {padding_bottom}%;">
    <iframe
        src="{embed_url}"
        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
        frameborder="0"
        scrolling="no"
        allowfullscreen="true">
    </iframe>
</div>
"""

        return responsive_html.strip()

    def generate_javascript_embed_code(
        self,
        embed_config: EmbedConfig,
        container_id: str = "dashboard-container",
        additional_params: Optional[dict[str, str]] = None,
    ) -> str:
        """Generate JavaScript embed code.

        Args:
            embed_config: Embed configuration
            container_id: Container element ID
            additional_params: Additional URL parameters

        Returns:
            JavaScript embed code
        """
        embed_url = self._build_embed_url(embed_config, additional_params)

        js_code = f"""
<div id="{container_id}"></div>
<script>
(function() {{
    var container = document.getElementById('{container_id}');
    if (!container) {{
        console.error('Dashboard container not found: {container_id}');
        return;
    }}

    var iframe = document.createElement('iframe');
    iframe.src = '{embed_url}';
    iframe.width = '{embed_config.width or "100%"}';
    iframe.height = '{embed_config.height or 600}';
    iframe.frameBorder = '0';
    iframe.scrolling = 'no';
    iframe.allowFullscreen = true;

    // Add error handling
    iframe.onerror = function() {{
        container.innerHTML = '<p>Failed to load dashboard</p>';
    }};

    container.appendChild(iframe);
}})();
</script>
"""

        return js_code.strip()

    def _build_embed_url(
        self,
        embed_config: EmbedConfig,
        additional_params: Optional[dict[str, str]] = None,
    ) -> str:
        """Build embed URL with parameters.

        Args:
            embed_config: Embed configuration
            additional_params: Additional URL parameters

        Returns:
            Complete embed URL
        """
        base_url = self.embed_manager.base_url

        # Build URL parameters
        params = {
            "embed_id": embed_config.id,
            "auto_refresh": str(embed_config.auto_refresh).lower(),
            "show_toolbar": str(embed_config.show_toolbar).lower(),
            "show_filters": str(embed_config.show_filters).lower(),
        }

        if additional_params:
            params.update(additional_params)

        # Build complete URL
        query_string = urlencode(params)
        embed_url = f"{base_url}/embed/dashboard/{embed_config.dashboard_id}?{query_string}"

        return embed_url


class APIEndpointGenerator:
    """Generates API endpoints for dashboard data access."""

    def __init__(self, embed_manager: EmbedManager):
        """Initialize API endpoint generator.

        Args:
            embed_manager: Embed manager instance
        """
        self.embed_manager = embed_manager

    def generate_dashboard_api_url(
        self, embed_config: EmbedConfig, format: str = "json", include_data: bool = True
    ) -> str:
        """Generate API URL for dashboard data.

        Args:
            embed_config: Embed configuration
            format: Response format (json, csv, etc.)
            include_data: Include widget data

        Returns:
            API endpoint URL
        """
        base_url = self.embed_manager.base_url

        params = {
            "embed_id": embed_config.id,
            "format": format,
            "include_data": str(include_data).lower(),
        }

        query_string = urlencode(params)
        api_url = f"{base_url}/api/dashboard/{embed_config.dashboard_id}/export?{query_string}"

        return api_url

    def generate_widget_api_url(
        self, embed_config: EmbedConfig, widget_id: str, format: str = "json"
    ) -> str:
        """Generate API URL for specific widget data.

        Args:
            embed_config: Embed configuration
            widget_id: Widget ID
            format: Response format

        Returns:
            Widget API endpoint URL
        """
        base_url = self.embed_manager.base_url

        params = {"embed_id": embed_config.id, "format": format}

        query_string = urlencode(params)
        api_url = f"{base_url}/api/dashboard/{embed_config.dashboard_id}/widget/{widget_id}?{query_string}"

        return api_url

    def generate_curl_example(
        self, embed_config: EmbedConfig, endpoint_type: str = "dashboard"
    ) -> str:
        """Generate curl command example.

        Args:
            embed_config: Embed configuration
            endpoint_type: Type of endpoint (dashboard, widget)

        Returns:
            Curl command example
        """
        if endpoint_type == "dashboard":
            url = self.generate_dashboard_api_url(embed_config)
        else:
            url = self.generate_widget_api_url(embed_config, "example-widget-id")

        curl_cmd = f'curl -X GET "{url}"'

        if embed_config.password_protected:
            curl_cmd += f' -H "Authorization: Bearer {embed_config.password}"'

        return curl_cmd

    def generate_python_example(
        self, embed_config: EmbedConfig, endpoint_type: str = "dashboard"
    ) -> str:
        """Generate Python code example.

        Args:
            embed_config: Embed configuration
            endpoint_type: Type of endpoint

        Returns:
            Python code example
        """
        if endpoint_type == "dashboard":
            url = self.generate_dashboard_api_url(embed_config)
        else:
            url = self.generate_widget_api_url(embed_config, "example-widget-id")

        python_code = f"""
import requests

url = "{url}"
"""

        if embed_config.password_protected:
            python_code += f"""
headers = {{"Authorization": "Bearer {embed_config.password}"}}
response = requests.get(url, headers=headers)
"""
        else:
            python_code += """
response = requests.get(url)
"""

        python_code += """
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}")
"""

        return python_code.strip()

    def generate_javascript_example(
        self, embed_config: EmbedConfig, endpoint_type: str = "dashboard"
    ) -> str:
        """Generate JavaScript code example.

        Args:
            embed_config: Embed configuration
            endpoint_type: Type of endpoint

        Returns:
            JavaScript code example
        """
        if endpoint_type == "dashboard":
            url = self.generate_dashboard_api_url(embed_config)
        else:
            url = self.generate_widget_api_url(embed_config, "example-widget-id")

        js_code = f"""
const url = "{url}";
"""

        if embed_config.password_protected:
            js_code += f"""
const headers = {{
    "Authorization": "Bearer {embed_config.password}"
}};

fetch(url, {{ headers }})
"""
        else:
            js_code += """
fetch(url)
"""

        js_code += """
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));
"""

        return js_code.strip()


class EmbedSecurityManager:
    """Manages security aspects of dashboard embedding."""

    def __init__(self):
        """Initialize embed security manager."""
        self._security_policies = {
            "max_embed_age_days": 365,
            "require_https": True,
            "allow_clickjacking": False,
            "max_embeds_per_dashboard": 10,
            "rate_limit_per_hour": 1000,
        }

    def validate_embed_security(self, embed_config: EmbedConfig) -> list[str]:
        """Validate embed security configuration.

        Args:
            embed_config: Embed configuration to validate

        Returns:
            List of security warnings/issues
        """
        warnings = []

        # Check expiration
        if not embed_config.expires_at:
            warnings.append("Embed has no expiration date")
        elif embed_config.expires_at > datetime.now() + timedelta(
            days=self._security_policies["max_embed_age_days"]
        ):
            warnings.append(
                f"Embed expires after {self._security_policies['max_embed_age_days']} days"
            )

        # Check password protection for public embeds
        if embed_config.public and not embed_config.password_protected:
            warnings.append("Public embed without password protection")

        # Check domain restrictions
        if not embed_config.allowed_domains:
            warnings.append("No domain restrictions configured")
        elif "*" in embed_config.allowed_domains:
            warnings.append("Wildcard domain allowed (security risk)")

        return warnings

    def generate_security_headers(self, embed_config: EmbedConfig) -> dict[str, str]:
        """Generate security headers for embed response.

        Args:
            embed_config: Embed configuration

        Returns:
            Dictionary of security headers
        """
        headers = {}

        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "connect-src 'self'",
        ]

        if embed_config.allowed_domains:
            # Add allowed domains to frame-ancestors
            domains = " ".join(embed_config.allowed_domains)
            csp_directives.append(f"frame-ancestors {domains}")
        else:
            csp_directives.append("frame-ancestors 'none'")

        headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # X-Frame-Options (fallback for older browsers)
        if embed_config.allowed_domains:
            # Note: X-Frame-Options doesn't support multiple domains
            headers["X-Frame-Options"] = "SAMEORIGIN"
        else:
            headers["X-Frame-Options"] = "DENY"

        # Other security headers
        headers.update(
            {
                "X-Content-Type-Options": "nosniff",
                "X-XSS-Protection": "1; mode=block",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            }
        )

        # HTTPS enforcement
        if self._security_policies["require_https"]:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return headers

    def create_embed_token(self, embed_config: EmbedConfig) -> str:
        """Create secure embed token.

        Args:
            embed_config: Embed configuration

        Returns:
            Secure embed token
        """
        # Create token payload
        payload = {
            "embed_id": embed_config.id,
            "dashboard_id": embed_config.dashboard_id,
            "expires_at": embed_config.expires_at.isoformat() if embed_config.expires_at else None,
            "created_at": datetime.now().isoformat(),
        }

        # Encode payload
        payload_json = str(payload).encode("utf-8")
        token = base64.urlsafe_b64encode(payload_json).decode("utf-8")

        return token

    def validate_embed_token(self, token: str) -> tuple[bool, Optional[dict[str, Any]]]:
        """Validate embed token.

        Args:
            token: Embed token to validate

        Returns:
            Tuple of (is_valid, payload)
        """
        try:
            # Decode token
            payload_json = base64.urlsafe_b64decode(token.encode("utf-8"))
            payload = eval(
                payload_json.decode("utf-8")
            )  # Note: Use proper JSON parsing in production

            # Check expiration
            if payload.get("expires_at"):
                expires_at = datetime.fromisoformat(payload["expires_at"])
                if expires_at < datetime.now():
                    return False, None

            return True, payload

        except Exception as e:
            logger.error(f"Token validation failed: {str(e)}")
            return False, None
