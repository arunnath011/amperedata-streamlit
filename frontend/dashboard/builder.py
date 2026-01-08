"""Dashboard builder with drag-and-drop interface and layout management.

This module provides the core dashboard building functionality including
layout management, widget positioning, and rendering orchestration.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from .exceptions import DashboardError, LayoutError, RenderingError, WidgetError
from .models import (
    DashboardConfig,
    DashboardLayout,
    DashboardWidget,
    LayoutType,
    WidgetPosition,
    WidgetType,
)
from .widgets import BaseWidget, create_widget

logger = logging.getLogger(__name__)


class LayoutManager:
    """Manages dashboard layout and widget positioning."""

    def __init__(self, layout: DashboardLayout):
        """Initialize layout manager.

        Args:
            layout: Dashboard layout configuration
        """
        self.layout = layout
        self._grid_occupied = set()
        self._widgets_positions = {}

    def validate_layout(self) -> bool:
        """Validate layout configuration.

        Returns:
            True if layout is valid

        Raises:
            LayoutError: If layout is invalid
        """
        if self.layout.grid_size[0] <= 0 or self.layout.grid_size[1] <= 0:
            raise LayoutError("Grid size must be positive")

        if self.layout.gap < 0:
            raise LayoutError("Gap cannot be negative")

        if self.layout.padding < 0:
            raise LayoutError("Padding cannot be negative")

        return True

    def add_widget(self, widget: DashboardWidget) -> bool:
        """Add widget to layout.

        Args:
            widget: Widget to add

        Returns:
            True if widget was added successfully

        Raises:
            LayoutError: If widget cannot be placed
        """
        if not self.can_place_widget(widget.position):
            raise LayoutError(
                f"Cannot place widget {widget.id} at position "
                f"({widget.position.x}, {widget.position.y})"
            )

        # Mark grid cells as occupied
        self._mark_occupied(widget.position)
        self._widgets_positions[widget.id] = widget.position

        logger.info(f"Added widget {widget.id} to layout")
        return True

    def remove_widget(self, widget_id: str) -> bool:
        """Remove widget from layout.

        Args:
            widget_id: ID of widget to remove

        Returns:
            True if widget was removed
        """
        if widget_id not in self._widgets_positions:
            return False

        position = self._widgets_positions[widget_id]
        self._mark_free(position)
        del self._widgets_positions[widget_id]

        logger.info(f"Removed widget {widget_id} from layout")
        return True

    def move_widget(self, widget_id: str, new_position: WidgetPosition) -> bool:
        """Move widget to new position.

        Args:
            widget_id: ID of widget to move
            new_position: New widget position

        Returns:
            True if widget was moved successfully

        Raises:
            LayoutError: If widget cannot be moved to new position
        """
        if widget_id not in self._widgets_positions:
            raise LayoutError(f"Widget {widget_id} not found in layout")

        old_position = self._widgets_positions[widget_id]

        # Temporarily free old position
        self._mark_free(old_position)

        # Check if new position is available
        if not self.can_place_widget(new_position):
            # Restore old position
            self._mark_occupied(old_position)
            raise LayoutError(
                f"Cannot move widget {widget_id} to position "
                f"({new_position.x}, {new_position.y})"
            )

        # Place widget in new position
        self._mark_occupied(new_position)
        self._widgets_positions[widget_id] = new_position

        logger.info(f"Moved widget {widget_id} to ({new_position.x}, {new_position.y})")
        return True

    def resize_widget(self, widget_id: str, new_size: tuple[int, int]) -> bool:
        """Resize widget.

        Args:
            widget_id: ID of widget to resize
            new_size: New widget size (width, height)

        Returns:
            True if widget was resized successfully

        Raises:
            LayoutError: If widget cannot be resized
        """
        if widget_id not in self._widgets_positions:
            raise LayoutError(f"Widget {widget_id} not found in layout")

        old_position = self._widgets_positions[widget_id]
        new_position = WidgetPosition(
            x=old_position.x,
            y=old_position.y,
            width=new_size[0],
            height=new_size[1],
            z_index=old_position.z_index,
        )

        # Temporarily free old position
        self._mark_free(old_position)

        # Check if new size fits
        if not self.can_place_widget(new_position):
            # Restore old position
            self._mark_occupied(old_position)
            raise LayoutError(f"Cannot resize widget {widget_id} to size {new_size}")

        # Apply new size
        self._mark_occupied(new_position)
        self._widgets_positions[widget_id] = new_position

        logger.info(f"Resized widget {widget_id} to {new_size}")
        return True

    def can_place_widget(self, position: WidgetPosition) -> bool:
        """Check if widget can be placed at position.

        Args:
            position: Widget position to check

        Returns:
            True if position is available
        """
        # Check bounds
        if (
            position.x < 0
            or position.y < 0
            or position.x + position.width > self.layout.grid_size[0]
            or position.y + position.height > self.layout.grid_size[1]
        ):
            return False

        # Check for overlaps
        for x in range(position.x, position.x + position.width):
            for y in range(position.y, position.y + position.height):
                if (x, y) in self._grid_occupied:
                    return False

        return True

    def find_available_position(self, width: int, height: int) -> Optional[WidgetPosition]:
        """Find available position for widget of given size.

        Args:
            width: Widget width
            height: Widget height

        Returns:
            Available position or None if no space
        """
        for y in range(self.layout.grid_size[1] - height + 1):
            for x in range(self.layout.grid_size[0] - width + 1):
                position = WidgetPosition(x=x, y=y, width=width, height=height)
                if self.can_place_widget(position):
                    return position

        return None

    def get_layout_utilization(self) -> float:
        """Get layout utilization percentage.

        Returns:
            Utilization percentage (0.0 to 1.0)
        """
        total_cells = self.layout.grid_size[0] * self.layout.grid_size[1]
        occupied_cells = len(self._grid_occupied)
        return occupied_cells / total_cells if total_cells > 0 else 0.0

    def optimize_layout(self) -> bool:
        """Optimize layout by compacting widgets.

        Returns:
            True if layout was optimized
        """
        # Simple optimization: move widgets up and left when possible
        widgets_by_position = sorted(
            self._widgets_positions.items(), key=lambda x: (x[1].y, x[1].x)
        )

        optimized = False
        for widget_id, position in widgets_by_position:
            # Try to move widget up
            for new_y in range(position.y):
                new_position = WidgetPosition(
                    x=position.x,
                    y=new_y,
                    width=position.width,
                    height=position.height,
                    z_index=position.z_index,
                )

                if self.can_place_widget_excluding(new_position, widget_id):
                    self.move_widget(widget_id, new_position)
                    optimized = True
                    break

        return optimized

    def can_place_widget_excluding(self, position: WidgetPosition, exclude_widget_id: str) -> bool:
        """Check if widget can be placed excluding specific widget.

        Args:
            position: Position to check
            exclude_widget_id: Widget ID to exclude from collision check

        Returns:
            True if position is available
        """
        # Temporarily remove excluded widget
        if exclude_widget_id in self._widgets_positions:
            excluded_position = self._widgets_positions[exclude_widget_id]
            self._mark_free(excluded_position)

            result = self.can_place_widget(position)

            # Restore excluded widget
            self._mark_occupied(excluded_position)
            return result

        return self.can_place_widget(position)

    def _mark_occupied(self, position: WidgetPosition) -> None:
        """Mark grid cells as occupied."""
        for x in range(position.x, position.x + position.width):
            for y in range(position.y, position.y + position.height):
                self._grid_occupied.add((x, y))

    def _mark_free(self, position: WidgetPosition) -> None:
        """Mark grid cells as free."""
        for x in range(position.x, position.x + position.width):
            for y in range(position.y, position.y + position.height):
                self._grid_occupied.discard((x, y))


class WidgetManager:
    """Manages dashboard widgets and their lifecycle."""

    def __init__(self):
        """Initialize widget manager."""
        self._widgets: dict[str, BaseWidget] = {}
        self._widget_configs: dict[str, DashboardWidget] = {}

    def add_widget(self, widget_config: DashboardWidget) -> BaseWidget:
        """Add widget to manager.

        Args:
            widget_config: Widget configuration

        Returns:
            Created widget instance

        Raises:
            WidgetError: If widget creation fails
        """
        try:
            widget = create_widget(widget_config.type, widget_config.id, widget_config.config)

            self._widgets[widget_config.id] = widget
            self._widget_configs[widget_config.id] = widget_config

            logger.info(f"Added widget {widget_config.id} of type {widget_config.type}")
            return widget

        except Exception as e:
            logger.error(f"Failed to add widget {widget_config.id}: {str(e)}")
            raise WidgetError(f"Widget creation failed: {str(e)}") from e

    def remove_widget(self, widget_id: str) -> bool:
        """Remove widget from manager.

        Args:
            widget_id: ID of widget to remove

        Returns:
            True if widget was removed
        """
        if widget_id in self._widgets:
            del self._widgets[widget_id]
            del self._widget_configs[widget_id]
            logger.info(f"Removed widget {widget_id}")
            return True

        return False

    def get_widget(self, widget_id: str) -> Optional[BaseWidget]:
        """Get widget by ID.

        Args:
            widget_id: Widget ID

        Returns:
            Widget instance or None
        """
        return self._widgets.get(widget_id)

    def get_widget_config(self, widget_id: str) -> Optional[DashboardWidget]:
        """Get widget configuration by ID.

        Args:
            widget_id: Widget ID

        Returns:
            Widget configuration or None
        """
        return self._widget_configs.get(widget_id)

    def update_widget_config(self, widget_id: str, config: DashboardWidget) -> bool:
        """Update widget configuration.

        Args:
            widget_id: Widget ID
            config: New widget configuration

        Returns:
            True if widget was updated

        Raises:
            WidgetError: If update fails
        """
        if widget_id not in self._widgets:
            raise WidgetError(f"Widget {widget_id} not found")

        try:
            # Create new widget with updated config
            new_widget = create_widget(config.type, widget_id, config.config)

            # Replace old widget
            self._widgets[widget_id] = new_widget
            self._widget_configs[widget_id] = config

            logger.info(f"Updated widget {widget_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update widget {widget_id}: {str(e)}")
            raise WidgetError(f"Widget update failed: {str(e)}") from e

    def refresh_widget_data(self, widget_id: str) -> bool:
        """Refresh data for specific widget.

        Args:
            widget_id: Widget ID

        Returns:
            True if data was refreshed
        """
        widget = self._widgets.get(widget_id)
        if not widget:
            return False

        try:
            widget.refresh_data()
            return True
        except Exception as e:
            logger.error(f"Failed to refresh data for widget {widget_id}: {str(e)}")
            return False

    def refresh_all_widgets(self) -> dict[str, bool]:
        """Refresh data for all widgets.

        Returns:
            Dictionary mapping widget IDs to refresh success status
        """
        results = {}
        for widget_id in self._widgets:
            results[widget_id] = self.refresh_widget_data(widget_id)

        return results

    def get_all_widgets(self) -> dict[str, BaseWidget]:
        """Get all widgets.

        Returns:
            Dictionary mapping widget IDs to widget instances
        """
        return self._widgets.copy()

    def get_widgets_by_type(self, widget_type: WidgetType) -> dict[str, BaseWidget]:
        """Get widgets by type.

        Args:
            widget_type: Widget type to filter by

        Returns:
            Dictionary mapping widget IDs to widget instances
        """
        return {
            widget_id: widget
            for widget_id, widget in self._widgets.items()
            if self._widget_configs[widget_id].type == widget_type
        }


class DashboardRenderer:
    """Renders dashboard layouts and widgets."""

    def __init__(self):
        """Initialize dashboard renderer."""
        self.render_cache = {}

    def render_dashboard(self, config: DashboardConfig) -> dict[str, Any]:
        """Render complete dashboard.

        Args:
            config: Dashboard configuration

        Returns:
            Rendered dashboard data

        Raises:
            RenderingError: If rendering fails
        """
        try:
            # Create layout manager
            layout_manager = LayoutManager(config.layout)
            layout_manager.validate_layout()

            # Create widget manager
            widget_manager = WidgetManager()

            # Add widgets to managers
            rendered_widgets = []
            for widget_config in config.widgets:
                try:
                    # Add to layout
                    layout_manager.add_widget(widget_config)

                    # Add to widget manager
                    widget = widget_manager.add_widget(widget_config)

                    # Render widget
                    rendered_widget = self.render_widget(widget, widget_config)
                    rendered_widgets.append(rendered_widget)

                except Exception as e:
                    logger.error(f"Failed to render widget {widget_config.id}: {str(e)}")
                    # Continue with other widgets
                    rendered_widgets.append(
                        {
                            "widget_id": widget_config.id,
                            "type": "error",
                            "error": str(e),
                            "position": widget_config.position.dict(),
                        }
                    )

            # Calculate layout metrics
            utilization = layout_manager.get_layout_utilization()

            dashboard_data = {
                "id": config.id,
                "name": config.name,
                "description": config.description,
                "layout": {
                    "type": config.layout.type,
                    "grid_size": config.layout.grid_size,
                    "gap": config.layout.gap,
                    "padding": config.layout.padding,
                    "responsive": config.layout.responsive,
                    "breakpoints": config.layout.breakpoints,
                    "utilization": utilization,
                },
                "widgets": rendered_widgets,
                "theme": config.theme,
                "auto_refresh": config.auto_refresh,
                "refresh_interval": config.refresh_interval,
                "rendered_at": datetime.now().isoformat(),
                "version": config.version,
            }

            logger.info(f"Rendered dashboard {config.id} with {len(rendered_widgets)} widgets")
            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to render dashboard {config.id}: {str(e)}")
            raise RenderingError(f"Dashboard rendering failed: {str(e)}") from e

    def render_widget(self, widget: BaseWidget, config: DashboardWidget) -> dict[str, Any]:
        """Render individual widget.

        Args:
            widget: Widget instance
            config: Widget configuration

        Returns:
            Rendered widget data
        """
        try:
            # Render widget content
            widget_data = widget.render()

            # Add position and styling information
            widget_data.update(
                {
                    "position": {
                        "x": config.position.x,
                        "y": config.position.y,
                        "width": config.position.width,
                        "height": config.position.height,
                        "z_index": config.position.z_index,
                    },
                    "style": config.style.dict(exclude_none=True),
                    "created_at": config.created_at.isoformat(),
                    "updated_at": config.updated_at.isoformat(),
                }
            )

            return widget_data

        except Exception as e:
            logger.error(f"Failed to render widget {widget.widget_id}: {str(e)}")
            return {
                "widget_id": widget.widget_id,
                "type": "error",
                "error": str(e),
                "position": config.position.dict(),
            }

    def render_widget_preview(
        self, widget_type: WidgetType, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Render widget preview for builder interface.

        Args:
            widget_type: Type of widget
            config: Widget configuration

        Returns:
            Widget preview data
        """
        try:
            # Create temporary widget
            temp_widget = create_widget(widget_type, "preview", config)

            # Render with minimal data
            preview_data = temp_widget.render()
            preview_data["is_preview"] = True

            return preview_data

        except Exception as e:
            logger.error(f"Failed to render widget preview: {str(e)}")
            return {"type": "error", "error": str(e), "is_preview": True}


class DashboardBuilder:
    """Main dashboard builder class with drag-and-drop functionality."""

    def __init__(self):
        """Initialize dashboard builder."""
        self.layout_manager: Optional[LayoutManager] = None
        self.widget_manager = WidgetManager()
        self.renderer = DashboardRenderer()
        self._current_config: Optional[DashboardConfig] = None

    def create_dashboard(
        self,
        name: str,
        description: Optional[str] = None,
        layout_type: LayoutType = LayoutType.GRID,
        grid_size: tuple[int, int] = (12, 12),
        created_by: str = "system",
    ) -> DashboardConfig:
        """Create new dashboard.

        Args:
            name: Dashboard name
            description: Dashboard description
            layout_type: Layout type
            grid_size: Grid dimensions
            created_by: Creator user ID

        Returns:
            Dashboard configuration
        """
        layout = DashboardLayout(type=layout_type, grid_size=grid_size)

        config = DashboardConfig(
            name=name, description=description, layout=layout, created_by=created_by
        )

        self._current_config = config
        self.layout_manager = LayoutManager(layout)

        logger.info(f"Created dashboard {config.id}: {name}")
        return config

    def load_dashboard(self, config: DashboardConfig) -> None:
        """Load existing dashboard configuration.

        Args:
            config: Dashboard configuration to load
        """
        self._current_config = config
        self.layout_manager = LayoutManager(config.layout)

        # Load widgets
        for widget_config in config.widgets:
            try:
                self.layout_manager.add_widget(widget_config)
                self.widget_manager.add_widget(widget_config)
            except Exception as e:
                logger.error(f"Failed to load widget {widget_config.id}: {str(e)}")

        logger.info(f"Loaded dashboard {config.id}")

    def add_widget(
        self,
        widget_type: WidgetType,
        config: dict[str, Any],
        position: Optional[WidgetPosition] = None,
        auto_position: bool = True,
    ) -> str:
        """Add widget to dashboard.

        Args:
            widget_type: Type of widget to add
            config: Widget configuration
            position: Widget position (optional)
            auto_position: Automatically find position if not specified

        Returns:
            Widget ID

        Raises:
            DashboardError: If widget cannot be added
        """
        if not self._current_config or not self.layout_manager:
            raise DashboardError("No dashboard loaded")

        # Create widget configuration
        widget_config = DashboardWidget(
            type=widget_type,
            position=position or WidgetPosition(x=0, y=0, width=4, height=3),
            config=config,
        )

        # Find position if needed
        if not position and auto_position:
            auto_pos = self.layout_manager.find_available_position(
                widget_config.position.width, widget_config.position.height
            )
            if auto_pos:
                widget_config.position = auto_pos
            else:
                raise DashboardError("No available space for widget")

        # Add to managers
        self.layout_manager.add_widget(widget_config)
        self.widget_manager.add_widget(widget_config)

        # Add to dashboard config
        self._current_config.widgets.append(widget_config)
        self._current_config.updated_at = datetime.now()

        logger.info(f"Added widget {widget_config.id} to dashboard")
        return widget_config.id

    def remove_widget(self, widget_id: str) -> bool:
        """Remove widget from dashboard.

        Args:
            widget_id: ID of widget to remove

        Returns:
            True if widget was removed
        """
        if not self._current_config or not self.layout_manager:
            return False

        # Remove from managers
        layout_removed = self.layout_manager.remove_widget(widget_id)
        widget_removed = self.widget_manager.remove_widget(widget_id)

        # Remove from config
        self._current_config.widgets = [
            w for w in self._current_config.widgets if w.id != widget_id
        ]
        self._current_config.updated_at = datetime.now()

        return layout_removed and widget_removed

    def move_widget(self, widget_id: str, new_position: WidgetPosition) -> bool:
        """Move widget to new position.

        Args:
            widget_id: ID of widget to move
            new_position: New position

        Returns:
            True if widget was moved
        """
        if not self.layout_manager:
            return False

        success = self.layout_manager.move_widget(widget_id, new_position)

        if success and self._current_config:
            # Update config
            for widget in self._current_config.widgets:
                if widget.id == widget_id:
                    widget.position = new_position
                    widget.updated_at = datetime.now()
                    break

            self._current_config.updated_at = datetime.now()

        return success

    def resize_widget(self, widget_id: str, new_size: tuple[int, int]) -> bool:
        """Resize widget.

        Args:
            widget_id: ID of widget to resize
            new_size: New size (width, height)

        Returns:
            True if widget was resized
        """
        if not self.layout_manager:
            return False

        success = self.layout_manager.resize_widget(widget_id, new_size)

        if success and self._current_config:
            # Update config
            for widget in self._current_config.widgets:
                if widget.id == widget_id:
                    widget.position.width = new_size[0]
                    widget.position.height = new_size[1]
                    widget.updated_at = datetime.now()
                    break

            self._current_config.updated_at = datetime.now()

        return success

    def update_widget_config(self, widget_id: str, config: dict[str, Any]) -> bool:
        """Update widget configuration.

        Args:
            widget_id: Widget ID
            config: New configuration

        Returns:
            True if widget was updated
        """
        if not self._current_config:
            return False

        # Find widget in config
        widget_config = None
        for widget in self._current_config.widgets:
            if widget.id == widget_id:
                widget_config = widget
                break

        if not widget_config:
            return False

        # Update widget config
        widget_config.config = config
        widget_config.updated_at = datetime.now()

        # Update in widget manager
        success = self.widget_manager.update_widget_config(widget_id, widget_config)

        if success:
            self._current_config.updated_at = datetime.now()

        return success

    def get_dashboard_config(self) -> Optional[DashboardConfig]:
        """Get current dashboard configuration.

        Returns:
            Dashboard configuration or None
        """
        return self._current_config

    def render_dashboard(self) -> dict[str, Any]:
        """Render current dashboard.

        Returns:
            Rendered dashboard data

        Raises:
            DashboardError: If no dashboard is loaded
        """
        if not self._current_config:
            raise DashboardError("No dashboard loaded")

        return self.renderer.render_dashboard(self._current_config)

    def validate_dashboard(self) -> list[str]:
        """Validate current dashboard configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self._current_config:
            errors.append("No dashboard configuration loaded")
            return errors

        # Validate layout
        try:
            if self.layout_manager:
                self.layout_manager.validate_layout()
        except LayoutError as e:
            errors.append(f"Layout error: {str(e)}")

        # Validate widgets
        for widget_config in self._current_config.widgets:
            try:
                widget = self.widget_manager.get_widget(widget_config.id)
                if widget:
                    widget.validate_config()
            except WidgetError as e:
                errors.append(f"Widget {widget_config.id} error: {str(e)}")

        return errors

    def optimize_layout(self) -> bool:
        """Optimize dashboard layout.

        Returns:
            True if layout was optimized
        """
        if not self.layout_manager:
            return False

        return self.layout_manager.optimize_layout()

    def get_layout_stats(self) -> dict[str, Any]:
        """Get layout statistics.

        Returns:
            Layout statistics
        """
        if not self.layout_manager or not self._current_config:
            return {}

        return {
            "total_widgets": len(self._current_config.widgets),
            "layout_utilization": self.layout_manager.get_layout_utilization(),
            "grid_size": self.layout_manager.layout.grid_size,
            "widget_types": {
                widget_type.value: len(self.widget_manager.get_widgets_by_type(widget_type))
                for widget_type in WidgetType
            },
        }
