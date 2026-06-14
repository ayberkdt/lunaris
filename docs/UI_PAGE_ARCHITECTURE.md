# Lunaris Desktop Page Architecture

## Shared Shell

```text
MainWindow
  ApplicationBar
  StatusSummary
  VerticalSplitter
    Workspace
      NavigationRail
      PageStack
        PageShell
          PageHeader
          Section / Subsection / page-owned content
    ExecutionConsoleDock
```

The shell owns navigation, page scrolling, console resizing, and workspace
persistence. Pages own scientific state and page-local interactions.

## Navigation

### Mission Studio

1. Orbit Setup
2. Force Models
3. Propagation
4. Results & Export
5. Live Telemetry
6. Data & Files
7. Monte Carlo

### ST-LRPS Studio

1. Data
2. Training Setup
3. Training Monitor
4. Evaluation
5. Runtime Performance
6. Orbit Benchmark
7. Gravity Plots

Both studios use the same selected, hover, focus, badge, and section language.
Feature names and command behavior remain unchanged.

## Page Responsibilities

### Orbit Setup / Mission

Keep orbit definition, preview, and derived values in one page. The header owns
the primary mission action. Derived values use `MetricRow`; validation appears
as an `InlineNotice`, not a modal unless execution cannot proceed.

### Force Models

Use one section per physical model. Each model presents enabled state, summary,
and configuration action. Advanced model dialogs remain page-owned.

### Propagation

Timeline and integrator settings use aligned `FormGrid` rows. Solver and
spacecraft dialogs remain available through concise secondary actions.

### Results & Export

Separate destination/settings from generated artifacts. Empty output folders use
`EmptyState`; filters and refresh actions live in a `Toolbar`.

### Live Telemetry

Plots remain the visual priority. Controls use a compact toolbar and metrics use
one comparison row. Plot internals are not wrapped in nested cards.

### Monte Carlo

Keep Setup, Run, and Results tabs. Backend selection and compatibility messaging
remain intact. Run status uses badges/notices; result tables and metrics share
the global comparison components.

### Data

Split source locations, dataset inspection, and validation into sections.
Path fields use the same labeled row and browse-action pattern.

### Training

Use dataset/model/optimization sections followed by a sticky action bar. The
command preview is a subdued code surface, not a competing card.

### Evaluation

Use a compact source toolbar, metric summary row, plots, and artifact table.
Empty and error states use shared components.

## State and Persistence

Page scientific state continues through existing `get_data`, `load_data`,
`get_state`, `apply_state`, `to_dict`, and `apply_dict` APIs. Visual state stores:

- active page;
- splitter sizes;
- console collapsed state;
- page-specific view/filter selections already supported by the application.

The console defaults collapsed for new sessions. Existing session values win
when restored.

## Responsive Rules

- At wide widths, page content is centered within the maximum readable width.
- At compact widths, the content column expands and scrolls; controls do not
  overlap the navigation or console.
- The console is a splitter child and never overlays page content.
- Long paths elide or scroll inside their field; they do not force the shell
  wider.
- Page actions remain reachable by keyboard and retain visible focus.
