# Lunaris Desktop Design System

## Principles

Lunaris is engineering software: dense information should be calm, comparable,
and predictable. The interface uses flat graphite surfaces, one orbital-blue
accent, semantic colors only for state, and whitespace instead of decorative
containers.

## Token Ownership

Typed tokens live in `lunaris.ui_foundation.tokens`, the single, Qt-binding-neutral
source of truth shared by both desktop applications. `lunaris.ui.theme.tokens` is a
compatibility re-export only — do not define tokens there. See
[UI_THEME.md](UI_THEME.md#ownership) for the full SSOT / re-export map.

- `ColorTokens`: surfaces, text tiers, borders, accent, and semantic colors.
- `TypographyTokens`: font families, sizes, and weights.
- `SpacingTokens`: a 4 px-based spacing scale.
- `RadiusTokens`: compact radii for controls, sections, and shells.
- `ControlMetrics`: minimum heights and standard compact widths.
- `LayoutTokens`: page width, navigation width, shell margins, and console
  dimensions.
- `VisualizationTokens`: plot, orbit, Moon, force-model, grid, and axis roles.

`DESIGN_TOKENS` is the typed source of truth. `THEME` remains available as a
read-only-compatible dictionary facade for existing pages and third-party plot
adapters.

## Color Roles

| Role | Token | Value |
| --- | --- | --- |
| App canvas | `bg_space` | `#090C12` |
| Shell chrome | `bg_shell` | `#0F141C` |
| Section surface | `bg_card` | `#171E29` |
| Elevated surface | `bg_card_alt` | `#212B38` |
| Inset/read-only surface | `bg_inset` | `#0C1016` |
| Input surface | `bg_entry` | `#0E131B` |
| Console surface | `bg_log` | `#06090E` |
| Primary text | `fg_main` | `#EEF2F8` |
| Secondary text | `fg_soft` | `#BCC7D6` |
| Muted text | `fg_muted` | `#8C99AA` |
| Disabled text | `fg_disabled` | `#79848F` |
| Primary accent | `accent` | `#3B86FF` |
| Secondary accent | `secondary` | `#15D6A6` |
| Success | `success` | `#14D49E` |
| Warning | `warning` | `#F5B43C` |
| Error | `error` | `#FF5D6C` |
| Critical | `critical` | `#FF7088` |
| Information | `info` | `#4A9DFF` |

There are no decorative gradients. Semantic colors are used for statuses,
not for generic card decoration.

## Type Hierarchy

- Application title: 15 pt, semibold.
- Page title: 18 pt, bold.
- Section title: 11 pt, semibold.
- Body/control text: 10 pt.
- Helper text and metadata: 9 pt.
- Console: 9.5 pt monospace.

## Layout Contract

- Shell margin: 16 px; shell gap: 12 px.
- Navigation width: 216 px wide, 188 px compact.
- Readable page width: 1180 px maximum, centered when extra space exists.
- Page vertical gap: 16 px.
- Section padding: 16 px.
- Form rows align labels and fields; units are separate muted labels.
- A page has one obvious primary action. Secondary actions move to toolbars or
  overflow menus.

## Component Contract

- `PageShell`: owns page margins, scrolling, and optional content width.
- `PageHeader`: title, concise description, optional status, optional action.
- `Section`: one meaningful grouping with title and optional description.
- `Subsection`: lighter grouping inside a section.
- `FormGrid`: aligned label/control/unit rows.
- `LabeledField` and `UnitField`: reusable compact field compositions.
- `KeyValueList` and `MetricRow`: comparison-friendly read-only data.
- `StatusBadge`: short text plus semantic kind.
- `InlineNotice`: explanatory, warning, success, or error message.
- `ActionBar` and `Toolbar`: predictable action placement.
- `SegmentedControl`: compact mutually-exclusive mode selection.
- `EmptyState`: title, guidance, optional action.
- `ExecutionConsoleDock`: compact status strip plus expandable console.
- `CompactSearchField` and `OverflowMenuButton`: reusable toolbar controls.

## QSS Rules

The global stylesheet owns surfaces, typography, borders, focus states, control
metrics, semantic variants, navigation, and the console. Page code should prefer
object names or dynamic properties over inline QSS.

Local styles are acceptable only for:

- runtime-generated plot series colors;
- OpenGL material colors;
- third-party widgets that do not expose a usable property/palette API;
- one-off geometry that cannot be represented by layout metrics.

All local colors must still originate from a token.
