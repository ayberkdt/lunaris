"""
Global application stylesheet builder for the Lunar Graphite theme.

The interface aims for a calm, professional, engineering-oriented dark theme:
flat surfaces, subtle borders, restrained hover states, and a single primary
accent (orbital blue).  Gradients are deliberately reserved for the primary
"Run" action and the progress-bar chunk; every other surface is flat.

All colors are routed through the ``THEME`` / ``LOG_COLORS`` palettes so there
are no page-local hard-coded values here.
"""

from __future__ import annotations

from lunaris.ui_foundation.palette import with_alpha
from lunaris.ui_foundation.tokens import DESIGN_TOKENS


def build_app_stylesheet(
    theme: dict[str, str],
    log_colors: dict[str, str],
    density: str = "comfortable",
    *,
    spin_up_icon: str | None = None,
    spin_down_icon: str | None = None,
) -> str:
    """Return the full application QSS string for the given palettes.

    Parameters
    ----------
    theme:
        The ``THEME`` token dictionary (Qt widget colors).
    log_colors:
        The ``LOG_COLORS`` token dictionary (used for the console default text).
    density:
        ``"comfortable"`` (default) or ``"compact"``. Compact tightens control
        heights and vertical paddings for long working sessions on small screens
        without changing colors, radii, or the spacing scale used by layouts.
    spin_up_icon, spin_down_icon:
        Optional filesystem paths (forward-slashed) to chevron images for the
        spin-box / combo-box arrows. Passed as plain strings so this builder
        stays Qt-binding-neutral. When omitted, the steppers are styled but keep
        Qt's native arrow glyphs rather than emitting a broken ``url()``.
    """

    compact = str(density).lower() == "compact"

    # Derived translucent accent variants — computed once so the QSS below never
    # hard-codes a raw ``rgba(...)`` literal.
    acc = theme["accent"]
    acc_hover_border = theme["accent_deep"]   # calm, deep-blue hover edge
    acc_06 = with_alpha(acc, 0.06)            # faint nav hover wash
    acc_dim = theme["accent_dim"]             # ~0.12 tinted background
    acc_18 = with_alpha(acc, 0.18)
    acc_20 = with_alpha(acc, 0.20)
    acc_30 = with_alpha(acc, 0.30)
    acc_35 = with_alpha(acc, 0.35)
    acc_40 = with_alpha(acc, 0.40)

    # Per-section navigation accents. Each sidebar group gets its own hue so the
    # workspace areas read as distinct zones (Data = teal, Training = blue,
    # Analysis = violet) instead of one monochrome blue rail.
    sec_data = theme["secondary"]
    sec_data_wash = with_alpha(sec_data, 0.09)
    sec_data_fill = with_alpha(sec_data, 0.16)
    sec_train = theme["accent"]
    sec_train_wash = with_alpha(sec_train, 0.09)
    sec_train_fill = acc_dim
    sec_analysis = theme["tertiary"]
    sec_analysis_wash = with_alpha(sec_analysis, 0.09)
    sec_analysis_fill = theme["tertiary_dim"]

    # Semantic badge tints.
    info_bg, info_bd = with_alpha(theme["accent"], 0.12), with_alpha(theme["accent"], 0.35)
    ok_bg, ok_bd = with_alpha(theme["success"], 0.12), with_alpha(theme["success"], 0.32)
    err_bg, err_bd = with_alpha(theme["error"], 0.12), with_alpha(theme["error"], 0.30)
    warn_bg, warn_bd = with_alpha(theme["warning"], 0.12), with_alpha(theme["warning"], 0.32)
    inactive_bg = with_alpha(theme["inactive"], 0.12)
    inactive_bd = with_alpha(theme["inactive"], 0.30)
    metrics = DESIGN_TOKENS.controls
    layout = DESIGN_TOKENS.layout
    type_tokens = DESIGN_TOKENS.typography

    # Density-derived control metrics. Compact mode tightens heights/paddings;
    # comfortable mode keeps the standard tokens. Colors/radii are unchanged.
    input_min_h = metrics.compact_height if compact else metrics.minimum_height
    primary_min_h = metrics.minimum_height if compact else metrics.primary_height
    field_pad_v = 4 if compact else 7
    button_pad_v = 4 if compact else 7
    nav_pad_v = 7 if compact else 11

    # Spin-box / combo-box stepper chrome. Once the inputs are themed, Qt draws
    # no native arrow inside a styled spin box, so the up/down controls looked
    # broken. We render them as a full-height strip on the right (a larger,
    # themed click target — Fitts) and drop in chevron images when supplied.
    # The arrow rules are only emitted when real image paths are available, so a
    # missing-qtawesome fallback keeps Qt's native arrows instead of blanks.
    if spin_up_icon and spin_down_icon:
        _stepper_arrow_qss = f"""
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: url({spin_up_icon});
            width: 10px; height: 10px;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow,
        QComboBox::down-arrow {{
            image: url({spin_down_icon});
            width: 10px; height: 10px;
        }}"""
    else:
        _stepper_arrow_qss = ""

    stepper_qss = f"""
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 22px;
            border-left: 1px solid {theme['border_soft']};
            border-top-right-radius: 8px;
            background: {theme['bg_card_alt']};
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 22px;
            border-left: 1px solid {theme['border_soft']};
            border-bottom-right-radius: 8px;
            background: {theme['bg_card_alt']};
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background: {theme['bg_hover']};
        }}
        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
            background: {theme['bg_inset']};
        }}{_stepper_arrow_qss}"""

    return f"""
        /* GLOBAL FOUNDATION — flat space-black canvas */
        QMainWindow,
        QWidget#centralRoot {{
            background: {theme['bg_space']};
            color: {theme['fg_main']};
        }}
        QWidget {{
            background: transparent;
            color: {theme['fg_main']};
            font-family: {type_tokens.family_ui};
            font-size: {type_tokens.size_body_pt:g}pt;
        }}
        QLabel {{
            background: transparent;
        }}
        QWidget#contentRoot,
        QStackedWidget#pages,
        QScrollArea,
        QScrollArea > QWidget > QWidget {{
            background: transparent;
            border: none;
        }}
        QToolTip {{
            background: {theme['bg_card_alt']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border_strong']};
            padding: 6px 8px;
        }}

        QDialog {{
            background: {theme['bg_space']};
            color: {theme['fg_main']};
        }}
        QDialogButtonBox {{
            background: transparent;
            border-top: 1px solid {theme['border_soft']};
            padding-top: 10px;
        }}
        QLabel#dialogTitle {{
            color: {theme['fg_main']};
            font-size: 16pt;
            font-weight: 700;
        }}
        QLabel#dialogDescription {{
            color: {theme['fg_muted']};
            font-size: 9pt;
        }}

        /* MENUS */
        QMenuBar {{
            background: {theme['bg_shell']};
            color: {theme['fg_main']};
            border-bottom: 1px solid {theme['border_soft']};
            padding: 4px 8px;
        }}
        QMenuBar::item {{
            background: transparent;
            padding: 6px 12px;
            border-radius: 8px;
        }}
        QMenuBar::item:selected {{
            background: {acc_dim};
            color: {theme['fg_soft']};
        }}
        QMenu {{
            background: {theme['bg_card']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border']};
            padding: 6px;
        }}
        QMenu::item {{
            padding: 7px 22px 7px 12px;
            border-radius: 8px;
        }}
        QMenu::item:selected {{
            background: {acc_dim};
            color: {theme['fg_soft']};
        }}
        QMenu::separator {{
            height: 1px;
            background: {theme['border_soft']};
            margin: 6px 8px;
        }}

        /* CONTAINERS — flat shell surfaces, no gradients */
        QFrame#header {{
            background: {theme['bg_shell']};
            border: 1px solid {theme['border_soft']};
            border-radius: 12px;
        }}
        QWidget#logPanel {{
            background: {theme['bg_shell']};
            border: 1px solid {theme['border_soft']};
            border-radius: 12px;
        }}
        QFrame#logHeader {{
            background: {theme['bg_shell']};
            border: 1px solid {with_alpha(theme['accent'], 0.24)};
            border-radius: 10px;
        }}
        QWidget#logBody {{
            background: {theme['bg_shell']};
            border: none;
        }}
        QFrame#stateFrame {{
            background: {acc_dim};
            border: 1px solid {acc_20};
            border-radius: 8px;
            padding: 2px 8px;
        }}
        QFrame#toolbar {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border_soft']};
            border-radius: 10px;
        }}
        /* Header context chips — quiet, clickable summaries of the two mission
           settings worth seeing at all times (gravity model, output path). They
           replace the separate status ribbon: an elevated pill on the shell so
           they read as chips, not buttons, and carry a leading icon + value. */
        QPushButton#headerContextChip {{
            background: {theme['bg_card_alt']};
            border: 1px solid {theme['border_soft']};
            border-radius: {DESIGN_TOKENS.radii.pill}px;
            color: {theme['fg_soft']};
            font-weight: 600;
            padding: 4px 12px;
            min-height: {metrics.status_badge_height}px;
            text-align: left;
        }}
        QPushButton#headerContextChip:hover {{
            border-color: {acc_hover_border};
            background: {theme['bg_hover']};
            color: {theme['fg_main']};
        }}
        QPushButton#headerContextChip:pressed {{
            background: {theme['bg_inset']};
        }}

        /* TEXT */
        QLabel#title {{
            font-size: 15pt;
            font-weight: 700;
            color: {theme['fg_soft']};
            letter-spacing: 0.3px;
        }}
        QLabel#runState {{
            color: {theme['fg_soft']};
            font-weight: 600;
        }}
        QLabel#progressText {{
            color: {theme['fg_muted']};
            font-size: 9pt;
        }}
        QLabel#pageTitle {{
            color: {theme['fg_main']};
            font-size: {type_tokens.size_page_title_pt:g}pt;
            font-weight: {type_tokens.weight_bold};
        }}
        QLabel#pageDescription,
        QLabel#sectionDescription,
        QLabel#fieldHint,
        QLabel#emptyStateDescription {{
            color: {theme['fg_muted']};
            font-size: {type_tokens.size_caption_pt:g}pt;
        }}
        QLabel#panelTitle {{
            color: {theme['fg_main']};
            font-size: {type_tokens.size_subsection_pt:g}pt;
            font-weight: {type_tokens.weight_semibold};
        }}
        QLabel#sectionTitle,
        QLabel#emptyStateTitle {{
            color: {theme['fg_main']};
            font-size: {type_tokens.size_section_pt:g}pt;
            font-weight: {type_tokens.weight_semibold};
        }}
        QLabel#fieldLabel, QLabel#keyLabel, QLabel#metricLabel {{
            color: {theme['fg_muted']};
        }}
        /* Derived (display-only) form values: the label echoes the field's
           ghost styling so read-only state is announced at the label too. */
        QLabel#fieldLabel[derived="true"] {{
            font-style: italic;
        }}
        QLabel#fieldUnit {{
            color: {theme['fg_muted']};
            min-width: 40px;
        }}
        QLabel#valueLabel, QLabel#metricValue {{
            color: {theme['fg_main']};
            font-weight: {type_tokens.weight_semibold};
        }}
        /* Value flagged as off-nominal (e.g. a backend that fell back from the
           requested one) — color reinforces the text, never the sole signal. */
        QLabel#valueLabel[kind="warning"], QLabel#metricValue[kind="warning"] {{
            color: {theme['warning']};
        }}
        /* Compact caption/value pair used by page-local detail rows (data,
           results, batch, force pages). Formerly also drove the mission-status
           ribbon, which is gone; these remain in use by those pages. */
        QLabel#statusLabel {{
            color: {theme['fg_muted']};
            font-size: 9pt;
        }}
        QLabel#statusValue {{
            color: {theme['fg_soft']};
            font-size: 9pt;
            font-weight: 600;
        }}

        /* NAVIGATION — flat shell; active = left border + tinted background */
        QListWidget#navDrawer {{
            background: {theme['bg_shell']};
            border: 1px solid {theme['border_soft']};
            border-radius: 12px;
            padding: 10px;
            outline: none;
            max-width: {layout.nav_width}px;
        }}
        QListWidget#navDrawer::item {{
            background: transparent;
            padding: {nav_pad_v}px 14px;
            margin-bottom: 6px;
            border-radius: 10px;
            color: {theme['fg_muted']};
        }}
        QListWidget#navDrawer::item:hover {{
            background: {acc_06};
            color: {theme['fg_main']};
        }}
        QListWidget#navDrawer::item:selected {{
            background: {acc_dim};
            color: {theme['fg_main']};
            font-weight: 700;
            border-left: 3px solid {theme['accent']};
        }}
        QFrame#navSidebar {{
            background: {theme['bg_shell']};
            border: 1px solid {theme['border_soft']};
            border-radius: 12px;
        }}
        /* Section labels carry a colored rail so each workspace zone is
           immediately distinguishable and the sidebar feels structured. */
        QLabel#navSectionLabel {{
            color: {theme['fg_muted']};
            font-size: 9pt;
            font-weight: 800;
            letter-spacing: 1.4px;
            padding: 4px 10px 4px 9px;
            margin-top: 6px;
            border-left: 3px solid {theme['border_strong']};
        }}
        QLabel#navSectionLabel[section="data"] {{
            color: {sec_data}; border-left-color: {sec_data};
        }}
        QLabel#navSectionLabel[section="train"] {{
            color: {sec_train}; border-left-color: {sec_train};
        }}
        QLabel#navSectionLabel[section="analysis"] {{
            color: {sec_analysis}; border-left-color: {sec_analysis};
        }}
        QFrame#navGroup {{
            background: {with_alpha(theme['bg_card'], 0.35)};
            border: 1px solid {theme['border_soft']};
            border-radius: 10px;
        }}
        QPushButton#navButton {{
            text-align: left;
            color: {theme['fg_soft']};
            background: transparent;
            border: 1px solid transparent;
            border-left: 3px solid transparent;
            border-radius: 8px;
            min-height: 38px;
            padding: 7px 12px;
            font-weight: 500;
        }}
        QPushButton#navButton:hover {{
            color: {theme['fg_main']};
            background: {acc_06};
        }}
        QPushButton#navButton:checked {{
            color: {theme['fg_main']};
            background: {acc_dim};
            border-left: 3px solid {theme['accent']};
            font-weight: 600;
        }}
        /* Per-section active + hover accents (override the generic blue rail). */
        QPushButton#navButton[section="data"]:hover {{ background: {sec_data_wash}; }}
        QPushButton#navButton[section="data"]:checked {{
            background: {sec_data_fill}; border-left-color: {sec_data};
        }}
        QPushButton#navButton[section="train"]:hover {{ background: {sec_train_wash}; }}
        QPushButton#navButton[section="train"]:checked {{
            background: {sec_train_fill}; border-left-color: {sec_train};
        }}
        QPushButton#navButton[section="analysis"]:hover {{ background: {sec_analysis_wash}; }}
        QPushButton#navButton[section="analysis"]:checked {{
            background: {sec_analysis_fill}; border-left-color: {sec_analysis};
        }}

        /* INPUTS — clean, legible, subtle states */
        QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox, QDateTimeEdit, QSpinBox {{
            background: {theme['bg_entry']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            padding: {field_pad_v}px 10px;
            min-height: {input_min_h}px;
            selection-background-color: {theme['accent']};
            selection-color: {theme['bg_space']};
        }}
        QLineEdit#compactSearch {{
            min-height: {metrics.compact_height}px;
            padding: 3px 9px;
        }}
        QLineEdit:hover, QComboBox:hover, QPlainTextEdit:hover,
        QDoubleSpinBox:hover, QDateTimeEdit:hover, QSpinBox:hover {{
            border: 1px solid {acc_hover_border};
        }}
        QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus,
        QDoubleSpinBox:focus, QDateTimeEdit:focus, QSpinBox:focus {{
            border: 2px solid {theme['accent']};
            background: {theme['bg_card_alt']};
        }}
        QLineEdit:read-only:enabled, QPlainTextEdit:read-only:enabled {{
            color: {theme['fg_soft']};
            background: {theme['bg_inset']};
            border: 1px dashed {theme['border']};
        }}
        /* The :read-only:enabled variant outranks the generic read-only rule
           above (two pseudo-classes vs. one attribute) — without it ghost
           fields keep the dashed "editable but locked" border. */
        QLineEdit[ghost="true"],
        QLineEdit[ghost="true"]:read-only:enabled {{
            color: {theme['fg_muted']};
            background: {theme['bg_inset']};
            /* Transparent (not dashed) border: a derived, display-only value
               must not read as an editable input; the 1px box is kept so the
               field does not shift when input modes swap. */
            border: 1px solid transparent;
            font-style: italic;
        }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
        QDoubleSpinBox:disabled, QDateTimeEdit:disabled {{
            color: {theme['fg_disabled']};
            background: {theme['bg_inset']};
            border-color: {theme['border_soft']};
        }}
        QComboBox::drop-down,
        QDateTimeEdit::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox QAbstractItemView {{
            background: {theme['bg_card_alt']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border']};
            selection-background-color: {acc_dim};
        }}
        {stepper_qss}
        QProgressBar {{
            background: {theme['bg_entry']};
            border: 1px solid {theme['border']};
            border-radius: 4px;
            min-height: 12px;
            max-height: 16px;
            text-align: center;
        }}
        QProgressBar::chunk {{
            background: {theme['accent']};
            border-radius: 3px;
        }}
        QFrame#appHeaderCard {{
            background: {theme['bg_shell']};
            border: 1px solid {theme['border_soft']};
            border-radius: 10px;
        }}
        QStatusBar {{
            background: {theme['bg_shell']};
            color: {theme['fg_muted']};
            border-top: 1px solid {theme['border_soft']};
            min-height: 22px;
        }}
        QStatusBar::item {{ border: none; }}

        /* CARDS — subtle border + surface contrast */
        QGroupBox {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border_soft']};
            border-radius: 12px;
            margin-top: 24px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            margin-left: 12px;
            color: {theme['fg_main']};
            font-size: 10.4pt;
            font-weight: 700;
        }}

        /* BUTTONS (default / secondary) — flat. A shared min-width keeps text
           actions from collapsing to a few pixels ("Fit", "Clear") so buttons
           read as one size class; icon-only and fixed-width chips override it. */
        QPushButton {{
            background: {theme['bg_card_alt']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            padding: {button_pad_v}px 16px;
            min-height: {input_min_h}px;
            min-width: 72px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            border-color: {acc_hover_border};
            background: {theme['bg_hover']};
            color: {theme['fg_main']};
        }}
        QPushButton:pressed {{
            background: {theme['bg_inset']};
        }}
        QPushButton:disabled {{
            background: {theme['bg_card']};
            border-color: {theme['border_soft']};
            color: {theme['fg_disabled']};
        }}
        QPushButton#quickChip {{
            background: {acc_dim};
            border: 1px solid {acc_20};
            color: {theme['fg_soft']};
            border-radius: 8px;
            padding: 5px 12px;
        }}
        QPushButton#quickChip:hover {{
            background: {acc_18};
            border-color: {theme['accent']};
        }}
        /* Icon-only square button (e.g. a settings gear). Overrides the shared
           text-button min-width so it stays a compact square target. */
        QPushButton#iconButton {{
            min-width: {metrics.icon_button_size}px;
            max-width: {metrics.icon_button_size}px;
            min-height: {metrics.icon_button_size}px;
            max-height: {metrics.icon_button_size}px;
            padding: 0;
        }}

        /* PRIMARY BUTTON (RUN) — the one place a gradient is welcome */
        QPushButton#primaryBtn,
        QPushButton[kind="primary"] {{
            background: {theme['accent']};
            border: 1px solid {theme['accent']};
            color: {theme['fg_inverse']};
            font-weight: 700;
            min-height: {primary_min_h}px;
        }}
        QPushButton#primaryBtn:hover,
        QPushButton[kind="primary"]:hover {{
            background: {theme['accent_hov']};
            border-color: {theme['accent_hov']};
            color: {theme['fg_inverse']};
        }}
        QPushButton#primaryBtn:disabled,
        QPushButton[kind="primary"]:disabled {{
            background: {theme['bg_entry']};
            border-color: {theme['border']};
            color: {theme['fg_disabled']};
        }}
        QPushButton[kind="ghost"] {{
            background: transparent;
            border-color: {theme['border_soft']};
            color: {theme['fg_soft']};
        }}
        QPushButton[kind="ghost"]:hover {{
            background: {theme['bg_hover']};
            border-color: {theme['border']};
            color: {theme['fg_main']};
        }}

        /* DANGER BUTTON (STOP) */
        QPushButton#dangerBtn,
        QPushButton[kind="danger"] {{
            background: {with_alpha(theme['error'], 0.10)};
            border: 1px solid {with_alpha(theme['error'], 0.26)};
            color: {theme['fg_main']};
        }}
        QPushButton#dangerBtn:hover,
        QPushButton[kind="danger"]:hover {{
            background: {with_alpha(theme['error'], 0.20)};
            border-color: {theme['error']};
        }}
        QPushButton#dangerBtn:disabled {{
            background: {theme['bg_entry']};
            border-color: {theme['border']};
            color: {theme['fg_disabled']};
        }}

        /* TOOL + CHECK CONTROLS */
        QToolButton {{
            background: transparent;
            border: none;
            padding: 4px;
        }}
        QToolButton:hover {{
            background: {acc_dim};
            border-radius: 8px;
        }}
        QToolButton#overflowMenuButton {{
            color: {theme['fg_soft']};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            padding: 5px 10px;
            min-height: {metrics.compact_height}px;
        }}
        QCheckBox {{
            spacing: 8px;
            color: {theme['fg_muted']};
        }}
        QCheckBox::indicator {{
            width: 14px;
            height: 14px;
            border-radius: 4px;
            border: 1px solid {theme['border']};
            background: {theme['bg_entry']};
        }}
        QCheckBox::indicator:checked {{
            background: {theme['accent']};
            border-color: {theme['accent']};
        }}
        QCheckBox::indicator:hover {{
            border-color: {acc_40};
        }}

        /* STATUS BADGE */
        QLabel#statusBadge {{
            border-radius: 10px;
            border: 1px solid {theme['border']};
            font-weight: 700;
            padding: 0 8px;
        }}
        QLabel#statusBadge[kind="info"] {{ background: {info_bg}; color: {theme['accent_hov']}; border-color: {info_bd}; }}
        QLabel#statusBadge[kind="success"] {{ background: {ok_bg}; color: {theme['success']}; border-color: {ok_bd}; }}
        QLabel#statusBadge[kind="error"] {{ background: {err_bg}; color: {theme['error']}; border-color: {err_bd}; }}
        QLabel#statusBadge[kind="warning"] {{ background: {warn_bg}; color: {theme['warning']}; border-color: {warn_bd}; }}
        QLabel#statusBadge[kind="idle"],
        QLabel#statusBadge[kind="inactive"],
        QLabel#statusBadge[kind="unavailable"],
        QLabel#statusBadge[kind="cancelled"] {{
            background: {inactive_bg};
            color: {theme['fg_muted']};
            border-color: {inactive_bd};
        }}
        QLabel#statusBadge[kind="validating"],
        QLabel#statusBadge[kind="running"],
        QLabel#statusBadge[kind="ready"] {{
            background: {info_bg};
            color: {theme['accent_hov']};
            border-color: {info_bd};
        }}
        QLabel#statusBadge[kind="completed"] {{
            background: {ok_bg};
            color: {theme['success']};
            border-color: {ok_bd};
        }}
        QLabel#statusBadge[kind="failed"],
        QLabel#statusBadge[kind="critical"] {{
            background: {err_bg};
            color: {theme['error']};
            border-color: {err_bd};
        }}
        QLabel#statusBadge[kind="paused"] {{
            background: {warn_bg};
            color: {theme['warning']};
            border-color: {warn_bd};
        }}

        /* PAGE PRIMITIVES */
        QWidget#pageShellBody {{
            background: transparent;
        }}
        QFrame#pageHeader {{
            background: transparent;
            border-bottom: 1px solid {theme['border_soft']};
            padding-bottom: 12px;
        }}
        QFrame#studioPageHeader {{
            background: transparent;
            border-bottom: 1px solid {theme['border_soft']};
        }}
        QLabel#pageEyebrow {{
            color: {theme['accent']};
            font-size: 9pt;
            font-weight: 700;
        }}
        QFrame[studioSurface="true"] {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border_soft']};
            border-radius: 10px;
        }}
        QFrame#section {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border_soft']};
            border-radius: 10px;
        }}
        QFrame#section[elevated="true"] {{
            background: {theme['bg_card_alt']};
            border-color: {theme['border']};
        }}
        QFrame#subsection {{
            background: transparent;
            border-top: 1px solid {theme['border_soft']};
        }}
        QFrame#metricCell {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border_soft']};
            border-radius: 8px;
        }}
        QFrame#orbitMetric,
        QFrame#metricCard {{
            background: {theme['bg_card_alt']};
            border: 1px solid {theme['border_soft']};
            border-radius: 8px;
        }}
        QLabel#orbitMetricLabel,
        QLabel#metricCardLabel {{
            color: {theme['fg_muted']};
            font-size: 9pt;
            font-weight: 600;
        }}
        QLabel#orbitMetricValue,
        QLabel#metricCardValue {{
            color: {theme['fg_main']};
            font-size: 12pt;
            font-weight: 700;
            font-family: {type_tokens.family_mono};
        }}
        /* Orbit preview: themed form rule + scene control chrome */
        QFrame#formDivider {{
            background: {theme['border_soft']};
            border: none;
            max-height: 1px;
            min-height: 1px;
        }}
        QLabel#orbitControlLabel {{
            color: {theme['fg_muted']};
            font-size: 8.5pt;
            font-weight: 700;
            letter-spacing: 0.6px;
        }}
        QPushButton#orbitPresetBtn {{
            background: {theme['bg_card_alt']};
            border: 1px solid {theme['border_soft']};
            border-radius: 7px;
            padding: 3px 10px;
            font-size: 9pt;
            font-weight: 600;
            color: {theme['fg_soft']};
        }}
        QPushButton#orbitPresetBtn:hover {{
            border-color: {acc_hover_border};
            background: {theme['bg_hover']};
            color: {theme['fg_main']};
        }}
        QPushButton#orbitPresetBtn:pressed {{
            background: {theme['bg_inset']};
        }}
        /* Orbit workspace split: transparent scroll surface + subtle handle */
        QScrollArea#orbitParamsScroll {{
            background: transparent;
            border: none;
        }}
        QScrollArea#orbitParamsScroll > QWidget > QWidget {{
            background: transparent;
        }}
        QSplitter#orbitSplit::handle {{
            background: transparent;
        }}
        QSplitter#orbitSplit::handle:hover {{
            background: {acc_20};
            border-radius: 3px;
        }}
        /* Live telemetry: cohesive control bar + idle empty-state overlay */
        QFrame#telemetryToolbar {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border_soft']};
            border-radius: 10px;
        }}
        /* "Scale" popover trigger — styled like a secondary button, not a bare
           tool button, so it sits consistently among the toolbar controls. */
        QToolButton#telemetryScaleButton {{
            background: {theme['bg_card_alt']};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            color: {theme['fg_soft']};
            padding: 4px 12px;
            font-weight: 600;
        }}
        QToolButton#telemetryScaleButton:hover {{
            border-color: {acc_hover_border};
            background: {theme['bg_hover']};
            color: {theme['fg_main']};
        }}
        QToolButton#telemetryScaleButton::menu-indicator {{
            image: none;
            width: 0px;
        }}
        QWidget#telemetryScalePanel {{
            background: {theme['bg_card']};
            min-width: 320px;
        }}
        QWidget#telemetryEmpty {{
            background: transparent;
        }}
        QLabel#telemetryEmptyTitle {{
            color: {theme['fg_soft']};
            font-size: 13pt;
            font-weight: 700;
        }}
        QLabel#telemetryEmptyText {{
            color: {theme['fg_muted']};
            font-size: 10pt;
        }}
        QFrame#telemetryKpiCell {{
            background: {theme['bg_card_alt']};
            border: 1px solid {theme['border_soft']};
            border-radius: 8px;
        }}
        QLabel#telemetryKpiLabel {{
            color: {theme['fg_muted']};
            font-size: 8.5pt;
            font-weight: 600;
        }}
        QLabel#telemetryKpiValue {{
            color: {theme['fg_main']};
            font-size: 13pt;
            font-weight: 700;
            font-family: {type_tokens.family_mono};
        }}
        QLabel#metricCardSubtitle {{
            color: {theme['fg_muted']};
            font-size: 9pt;
        }}
        QFrame#metricCard[state="success"] {{ border-color: {ok_bd}; background: {ok_bg}; }}
        QFrame#metricCard[state="warning"] {{ border-color: {warn_bd}; background: {warn_bg}; }}
        QFrame#metricCard[state="danger"] {{ border-color: {err_bd}; background: {err_bg}; }}
        QFrame#inlineNotice {{
            background: {info_bg};
            border: 1px solid {info_bd};
            border-radius: 8px;
        }}
        QFrame#inlineNotice[kind="success"] {{ background: {ok_bg}; border-color: {ok_bd}; }}
        QFrame#inlineNotice[kind="warning"] {{ background: {warn_bg}; border-color: {warn_bd}; }}
        QFrame#inlineNotice[kind="error"] {{ background: {err_bg}; border-color: {err_bd}; }}
        QLabel#inlineNoticeLabel {{
            color: {theme['fg_soft']};
            background: {info_bg};
            border: 1px solid {info_bd};
            border-radius: 8px;
            padding: 7px 10px;
        }}
        QLabel#inlineNoticeLabel[kind="ok"] {{ color: {theme['success']}; background: {ok_bg}; border-color: {ok_bd}; }}
        QLabel#inlineNoticeLabel[kind="warn"] {{ color: {theme['warning']}; background: {warn_bg}; border-color: {warn_bd}; }}
        QLabel#inlineNoticeLabel[kind="error"] {{ color: {theme['error']}; background: {err_bg}; border-color: {err_bd}; }}
        QFrame#actionBar {{
            background: transparent;
            border-top: 1px solid {theme['border_soft']};
        }}
        QFrame#segmentedControl,
        QWidget#segmentedControl {{
            background: {theme['bg_entry']};
            border: 1px solid {theme['border']};
            border-radius: 8px;
        }}
        QPushButton#segmentButton {{
            background: transparent;
            border: none;
            border-radius: 6px;
            min-height: {metrics.compact_height}px;
            padding: 4px 12px;
        }}
        QPushButton#segmentButton:checked {{
            background: {acc_dim};
            color: {theme['fg_main']};
        }}
        QFrame#emptyState {{
            background: {theme['bg_card']};
            border: 1px dashed {theme['border']};
            border-radius: 10px;
        }}

        /* ST-LRPS SHARED HEADER */
        QFrame#experimentHeader {{
            background: {theme['bg_shell']};
            border: 1px solid {theme['border_soft']};
            border-radius: 10px;
        }}
        QWidget#headerMetric {{
            background: {theme['bg_card_alt']};
            border: 1px solid {theme['border_soft']};
            border-radius: 8px;
        }}
        QLabel#headerMetricLabel {{
            color: {theme['fg_muted']};
            font-size: 8pt;
            font-weight: 700;
        }}
        QLabel#headerMetricValue {{
            color: {theme['fg_main']};
            font-size: 9pt;
            font-weight: 600;
        }}

        /* RUN DOT */
        QFrame#runDot {{ border-radius: 6px; }}
        QFrame#runDot[kind="idle"] {{ background: {theme['fg_muted']}; }}
        QFrame#runDot[kind="running"] {{ background: {theme['success']}; }}
        QFrame#runDot[kind="error"] {{ background: {theme['error']}; }}
        QFrame#runDot[kind="warning"] {{ background: {theme['warning']}; }}

        /* TABS */
        QTabWidget::pane {{
            border: 1px solid {theme['border_soft']};
            background: {theme['bg_card']};
            border-radius: 12px;
            top: -1px;
        }}
        QTabBar::tab {{
            background: {theme['bg_card_alt']};
            color: {theme['fg_muted']};
            border: 1px solid {theme['border_soft']};
            padding: 8px 16px;
            margin-right: 6px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }}
        QTabBar::tab:selected {{
            background: {acc_dim};
            color: {theme['fg_soft']};
            border-color: {acc_30};
            border-bottom: 2px solid {theme['accent']};
        }}
        QTabBar::tab:hover:!selected {{
            color: {theme['fg_main']};
            background: {theme['bg_entry']};
        }}

        /* LOGGING */
        QTextEdit#log {{
            background: {theme['bg_log']};
            color: {log_colors['default']};
            font-family: "Consolas", "Courier New", monospace;
            font-size: 10pt;
            border: 1px solid {theme['border_soft']};
            border-radius: 10px;
            padding: 8px;
        }}
        QPlainTextEdit#logConsole {{
            background: {theme['bg_log']};
            color: {log_colors['default']};
            border: 1px solid {theme['border_soft']};
            border-radius: 10px;
            padding: 11px 12px;
            font-family: {type_tokens.family_mono};
            font-size: 10pt;
            selection-background-color: {with_alpha(theme['accent'], 0.32)};
            selection-color: {theme['fg_main']};
        }}
        QPlainTextEdit#commandPreview {{
            background: {theme['bg_log']};
            border: 1px solid {acc_20};
            border-radius: 10px;
            color: {theme['fg_main']};
            padding: 10px 12px;
            font-family: {type_tokens.family_mono};
        }}
        QPlainTextEdit#commandPreview[state="error"] {{
            color: {theme['error']};
            border-color: {err_bd};
        }}
        QLabel#logTitle {{
            color: {theme['fg_main']};
            font-weight: 700;
        }}
        QLabel#logSubtitle, QLabel#logLatestMessage {{
            color: {theme['fg_soft']};
            font-size: 9pt;
        }}
        QLabel#logCounter {{
            color: {theme['fg_soft']};
            background: {theme['bg_entry']};
            border: 1px solid {theme['border_soft']};
            border-radius: 8px;
            padding: 2px 8px;
        }}
        QLabel#logStatusChip {{
            color: {theme['accent_hov']};
            background: {with_alpha(theme['accent'], 0.12)};
            border: 1px solid {with_alpha(theme['accent'], 0.28)};
            border-radius: 8px;
            padding: 2px 8px;
        }}
        QPushButton#logToolbarButton {{
            background: {theme['bg_entry']};
            border: 1px solid {theme['border_soft']};
            border-radius: 8px;
            color: {theme['fg_soft']};
            min-height: {metrics.compact_height}px;
            padding: 4px 10px;
        }}
        QPushButton#logToolbarButton:hover {{
            background: {theme['bg_hover']};
            border-color: {theme['accent_deep']};
            color: {theme['fg_main']};
        }}
        QToolButton#logCollapseButton {{
            color: {theme['fg_soft']};
            background: {theme['bg_entry']};
            border: 1px solid {theme['border_soft']};
            border-radius: 8px;
            font-weight: 700;
        }}
        QToolButton#logCollapseButton:hover {{
            color: {theme['fg_main']};
            background: {with_alpha(theme['accent'], 0.14)};
            border-color: {theme['accent_deep']};
        }}

        /* SCROLLBARS */
        QScrollBar:vertical, QScrollBar:horizontal {{
            background: transparent;
            width: 8px;
            height: 8px;
            margin: 0;
        }}
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
            background: {theme['border']};
            border-radius: 4px;
            min-height: 20px;
            min-width: 20px;
        }}
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
            background: {acc_40};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ height: 0px; width: 0px; }}

        /* SPLITTER */
        QSplitter::handle {{
            background: {theme['border_soft']};
        }}
        QSplitter::handle:vertical {{
            height: 8px;
            margin: 0 4px;
            border-radius: 3px;
            image: none;
        }}
        QSplitter::handle:vertical:hover {{
            background: {acc_35};
        }}
        QSplitter#mainSplit::handle:vertical {{
            background: {with_alpha(theme['accent'], 0.22)};
            height: 10px;
            border-radius: 4px;
        }}
        QSplitter#mainSplit::handle:vertical:hover {{
            background: {with_alpha(theme['accent'], 0.46)};
        }}

        /* ACCESSIBLE KEYBOARD FOCUS */
        QPushButton:focus, QToolButton:focus, QCheckBox:focus,
        QListWidget:focus, QTreeWidget:focus {{
            border: 1px solid {theme['accent']};
        }}

        /* TREE / LIST WIDGETS */
        QTreeWidget, QListWidget {{
            background: {theme['bg_entry']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border']};
            border-radius: 10px;
            outline: none;
            alternate-background-color: {theme['bg_card_alt']};
        }}
        QTreeWidget::item, QListWidget::item {{
            padding: 4px 6px;
            border-radius: 5px;
        }}
        QTreeWidget::item:selected, QListWidget::item:selected {{
            background: {acc_dim};
            color: {theme['fg_main']};
        }}
        QTreeWidget::item:hover, QListWidget::item:hover {{
            background: {acc_06};
        }}
        QHeaderView::section {{
            background: {theme['bg_card']};
            color: {theme['fg_soft']};
            border: none;
            border-bottom: 1px solid {theme['border']};
            padding: 5px 8px;
            font-weight: 600;
        }}

        /* DATA TABLES (results / ephemeris / event logs) */
        QTableWidget#dataTable {{
            background: {theme['bg_entry']};
            alternate-background-color: {theme['bg_card_alt']};
            color: {theme['fg_main']};
            gridline-color: {theme['border_soft']};
            border: 1px solid {theme['border']};
            border-radius: 10px;
            outline: none;
        }}
        QTableWidget#dataTable::item {{
            padding: 4px 8px;
        }}
        QTableWidget#dataTable::item:selected {{
            background: {acc_dim};
            color: {theme['fg_main']};
        }}
    """
