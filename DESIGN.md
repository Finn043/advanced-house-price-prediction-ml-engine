---
name: Melbourne Housing Model Studio
description: A calm, readable workspace for training and exploring a Melbourne housing model.
colors:
  primary: "#397354"
  primary-strong: "#2d6348"
  primary-wash: "#e9f1eb"
  page: "#f4f5f0"
  surface: "#ffffff"
  surface-soft: "#f8f9f6"
  ink: "#202820"
  muted: "#6d776e"
  line: "#e4e8e1"
  training-surface: "#273d30"
  chart: "#4b8863"
typography:
  body:
    fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  headline:
    fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "clamp(34px, 4vw, 49px)"
    fontWeight: 540
    lineHeight: 1.09
    letterSpacing: "-0.045em"
  title:
    fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "16px"
    fontWeight: 620
    lineHeight: 1.3
rounded:
  control: "7px"
  note: "8px"
  card: "13px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "40px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#ffffff"
    rounded: "{rounded.control}"
    padding: "0 16px"
    height: "40px"
  button-train:
    backgroundColor: "#dcebdd"
    textColor: "#263b2d"
    rounded: "{rounded.control}"
    padding: "0 16px"
    height: "42px"
  card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.card}"
    padding: "25px 28px"
  input:
    backgroundColor: "{colors.surface-soft}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "0 12px"
    height: "41px"
---

# Design System: Melbourne Housing Model Studio

## Overview

**Creative North Star: “The Learning Workbench” (working direction).**

The interface treats model training as hands-on work: property inputs and training controls sit together, followed by a learning curve that shows what the selected loss did over time. The visual tone is restrained and practical, with a muted green accent connecting the housing context to the model's training state. The light theme is the default; a dark theme keeps the same hierarchy with deeper green surfaces.

The page favors clear labels, compact controls, and explanatory notes over decorative imagery. Visual emphasis follows the user's sequence: describe a home, train the model, read the training curve, then request an estimate. This is an implementation-derived direction, not a separately confirmed brand identity.

**Key Characteristics:**
- Calm green accents over soft neutral surfaces.
- Clear separation between property entry and training controls.
- Training progress is shown as data, with its scope explained in plain language.

## Colors

The palette uses a restrained leaf green for actions and chart marks, balanced by warm, slightly green-tinted neutrals; dark mode deepens the same palette.

### Primary
- **Melbourne Leaf** (`{colors.primary}`): Main action, active navigation, section marks, and the learning-curve line.
- **Deep Leaf** (`{colors.primary-strong}`): Hover emphasis for primary actions.
- **Leaf Tint** (`{colors.primary-wash}`): Result and note surfaces that need a quiet connection to the primary green.

### Neutral
- **Soft Sage Paper** (`{colors.page}`): Main light-theme page background.
- **Clean White** (`{colors.surface}`): Primary cards and chart surface.
- **Pale Sage** (`{colors.surface-soft}`): Input and secondary surface fill.
- **Forest Ink** (`{colors.ink}`): Main text on light surfaces.
- **Muted Sage** (`{colors.muted}`): Secondary labels and helper text.
- **Quiet Divider** (`{colors.line}`): Card outlines and separators.
- **Deep Forest Panel** (`{colors.training-surface}`): Training-controls panel, retained as a dark anchor in either theme.

### Named Rules
**The Green Means Action Rule.** Reserve the strongest green for actions, selected states, and the chart; keep explanatory text neutral.

## Typography

**Display Font:** System UI sans-serif stack.
**Body Font:** System UI sans-serif stack.
**Label/Mono Font:** No separate mono face; numeric readouts use tabular figures.

**Character:** Neutral system typography keeps the dashboard familiar and quick to scan. Large headings are compact and lightly tracked; small labels remain plain rather than ornamental.

### Hierarchy
- **Headline** (weight 540, responsive 34–49px, line-height 1.09): Page headline, with tight negative tracking.
- **Title** (weight 620, 16px): Card and section headings.
- **Body** (weight 400, 14px): Introductory and explanatory copy, generally kept to a readable measure.
- **Label** (weight 450–600, 9–13px): Field labels, navigation, metadata, and chart annotations.

## Layout

The desktop workspace uses a centered content area capped at 1220px. The property form occupies the wider column and training controls the narrower column; the learning curve spans beneath both. The header aligns to the same centered width. Cards use consistent internal spacing and modest gaps rather than dense table-like layouts.

At 900px the content narrows and card spacing tightens. At 700px the workspace becomes a single column, the intro and learning summary stack, and the footer becomes vertical. At 420px property fields stack into one column; at 360px the secondary navigation is hidden to preserve room for the brand and theme control. Keep all controls usable without horizontal page scrolling.

## Elevation & Depth

Depth comes from tonal surfaces and fine borders, not drop shadows. The dark green training panel provides the main visual anchor; white cards and pale input fills establish hierarchy in light mode. Theme switching updates the page, cards, text, borders, and chart while preserving the training panel's contrasting role.

## Shapes

Cards use gently rounded corners (13px), controls are tighter (7px), and explanatory notes sit between those forms (8px). The theme control and status indicators are circular. Borders are thin and low contrast; avoid heavy outlines or pill-shaped form fields.

## Components

- **Property form:** Labels sit above numeric inputs and selects in a two-column grid. On narrow phones, fields become a single column. The estimate action spans the available width on mobile.
- **Training panel:** A dark green card groups loss selection, Batch GD/SGD choice, learning rate, epoch count, and the train action. Its pale action button contrasts against the panel.
- **Learning curve:** A bordered card displays loss by epoch as a responsive SVG line, with sparse gridlines and a labeled endpoint. Before training, retain the centered empty state and explain that training produces the curve.
- **Primary action:** Solid green with white text, a small directional icon, and a subtle lift on hover. Disabled actions reduce opacity and use a blocked cursor.
- **Inputs and selects:** Pale filled controls with a fine border, clear labels, and a green focus treatment. Preserve native keyboard and form behavior.
- **Prediction result:** A lightly tinted green inset keeps the AUD estimate prominent and includes the selected loss and optimizer as context.
- **Navigation and theme:** Simple text links use a green active underline. The circular theme button remains in the header and has an explicit accessible label.

## Do's and Don'ts

### Do
- **Do** keep the training-to-prediction sequence visible in the page hierarchy.
- **Do** pair every metric or chart with a clear label and explain that the current curve measures training data.
- **Do** preserve visible keyboard focus and readable contrast in both themes.
- **Do** let cards and form fields stack cleanly on small screens.

### Don't
- **Don't** style training loss as a measure of accuracy on unseen homes.
- **Don't** introduce bright unrelated accent colors or decorative imagery that competes with the data.
- **Don't** rely on color alone to communicate API status, selected controls, or loss direction.
- **Don't** add shadows or oversized rounded controls; the current hierarchy depends on tonal surfaces and quiet borders.
