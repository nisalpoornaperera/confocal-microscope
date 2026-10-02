/** Small presentational building blocks shared by the pages. */
import { useId, useState, type ChangeEvent, type ReactNode } from "react";

import type { ApiError } from "../api/client";
import type { PointStatus, ScanState } from "../api/types";
import { humanize } from "../lib/format";

export function Card({
  title,
  actions,
  children,
  className,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className ?? ""}`}>
      {(title !== undefined || actions !== undefined) && (
        <div className="card-header">
          {title !== undefined && <h2>{title}</h2>}
          {actions !== undefined && <div className="row">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="page-header">
      <h1>{title}</h1>
      {children !== undefined && <div className="row">{children}</div>}
    </div>
  );
}

/**
 * Display of a backend `ErrorResponse {error, detail, violations}` (or a
 * network failure), listing every violation.
 */
export function ErrorBox({
  error,
  title,
  onDismiss,
}: {
  error: ApiError | null | undefined;
  title?: string;
  onDismiss?: () => void;
}) {
  if (!error) return null;
  const heading =
    title ??
    (error.isNetworkError
      ? "The backend cannot be reached"
      : `${describeStatus(error.status)} (${error.error})`);
  return (
    <div className="alert alert-error" role="alert">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="alert-title">{heading}</div>
        {onDismiss && (
          <button type="button" className="btn-small" onClick={onDismiss}>
            Dismiss
          </button>
        )}
      </div>
      <div>{error.detail}</div>
      {error.violations.length > 0 && (
        <ul>
          {error.violations.map((violation, index) => (
            <li key={index}>{violation}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function describeStatus(status: number): string {
  switch (status) {
    case 404:
      return "Not found";
    case 409:
      return "Refused";
    case 422:
      return "Invalid request";
    case 503:
      return "Hardware failure";
    case 500:
      return "Server error";
    default:
      return `Error ${status}`;
  }
}

export function Alert({
  kind,
  title,
  children,
}: {
  kind: "info" | "warning" | "error" | "ok";
  title?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className={`alert alert-${kind}`} role={kind === "error" ? "alert" : "status"}>
      {title !== undefined && <div className="alert-title">{title}</div>}
      {children}
    </div>
  );
}

type BadgeKind = "ok" | "warn" | "danger" | "info" | "neutral";

export function Badge({ kind = "neutral", children, title }: { kind?: BadgeKind; children: ReactNode; title?: string }) {
  return (
    <span className={`badge ${kind === "neutral" ? "" : `badge-${kind}`}`} title={title}>
      {children}
    </span>
  );
}

export function scanStateKind(state: ScanState | null | undefined): BadgeKind {
  switch (state) {
    case "complete":
      return "ok";
    case "error":
      return "danger";
    case "cancelled":
      return "warn";
    case "paused":
      return "warn";
    case null:
    case undefined:
    case "idle":
      return "neutral";
    default:
      return "info";
  }
}

export function ScanStateBadge({ state, interrupted }: { state: ScanState; interrupted?: boolean }) {
  return (
    <span className="row" style={{ gap: "0.35rem", display: "inline-flex" }}>
      <Badge kind={scanStateKind(state)}>{humanize(state)}</Badge>
      {interrupted === true && (
        <Badge kind="warn" title="The scan ended before all points were measured">
          Interrupted
        </Badge>
      )}
    </span>
  );
}

export function pointStatusKind(status: PointStatus): BadgeKind {
  switch (status) {
    case "valid":
    case "measured":
      return "ok";
    case "low_confidence":
      return "warn";
    case "aborted":
    case "error":
      return "danger";
    default:
      return "warn";
  }
}

export function KeyValue({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="kv">
      {items.map(([key, value], index) => (
        <div key={index} style={{ display: "contents" }}>
          <dt>{key}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint !== undefined && <div className="small muted">{hint}</div>}
    </div>
  );
}

export function ProgressBar({ fraction, label }: { fraction: number; label: string }) {
  const clamped = Math.min(1, Math.max(0, Number.isFinite(fraction) ? fraction : 0));
  return (
    <div
      className="progress"
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped * 100)}
    >
      <div className="progress-bar" style={{ width: `${clamped * 100}%` }} />
    </div>
  );
}

export interface FieldProps {
  label: string;
  unit?: string;
  hint?: ReactNode;
  error?: string | null;
  children: (id: string, describedBy: string | undefined) => ReactNode;
}

/** Label + control + hint/error, wired up with ids for screen readers. */
export function Field({ label, unit, hint, error, children }: FieldProps) {
  const id = useId();
  const hintId = `${id}-hint`;
  const describedBy = error || hint !== undefined ? hintId : undefined;
  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>
        <span>{label}</span>
        {unit !== undefined && <span className="field-unit">{unit}</span>}
      </label>
      {children(id, describedBy)}
      {error ? (
        <div id={hintId} className="field-error">
          {error}
        </div>
      ) : hint !== undefined ? (
        <div id={hintId} className="field-hint">
          {hint}
        </div>
      ) : null}
    </div>
  );
}

/** Numeric text input that keeps what the user typed (validation happens on submit / estimate). */
export function NumberField({
  label,
  unit,
  hint,
  error,
  value,
  onChange,
  disabled,
  step = "any",
  min,
  max,
}: {
  label: string;
  unit?: string;
  hint?: ReactNode;
  error?: string | null;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  step?: number | "any";
  min?: number;
  max?: number;
}) {
  return (
    <Field label={label} unit={unit} hint={hint} error={error}>
      {(id, describedBy) => (
        <input
          id={id}
          type="number"
          inputMode="decimal"
          step={step}
          min={min}
          max={max}
          value={value}
          disabled={disabled}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          onChange={(event: ChangeEvent<HTMLInputElement>) => {
            onChange(event.target.value);
          }}
        />
      )}
    </Field>
  );
}

export function SelectField<T extends string>({
  label,
  hint,
  value,
  options,
  onChange,
  disabled,
}: {
  label: string;
  hint?: ReactNode;
  value: T;
  options: readonly (T | { value: T; label: string })[];
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <Field label={label} hint={hint}>
      {(id, describedBy) => (
        <select
          id={id}
          value={value}
          disabled={disabled}
          aria-describedby={describedBy}
          onChange={(event) => {
            onChange(event.target.value as T);
          }}
        >
          {options.map((option) => {
            const item = typeof option === "string" ? { value: option, label: humanize(option) } : option;
            return (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            );
          })}
        </select>
      )}
    </Field>
  );
}

export function CheckField({
  label,
  hint,
  checked,
  onChange,
  disabled,
}: {
  label: ReactNode;
  hint?: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="check">
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => {
          onChange(event.target.checked);
        }}
      />
      <span>
        {label}
        {hint !== undefined && <span className="field-hint" style={{ display: "block" }}>{hint}</span>}
      </span>
    </label>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  disabled,
}: {
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (value: T) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={option.value === value}
          disabled={disabled}
          onClick={() => {
            onChange(option.value);
          }}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/**
 * A button that asks for confirmation inline (no browser dialogs: kiosk
 * browsers may suppress them).
 */
export function ConfirmButton({
  children,
  confirmLabel,
  question,
  onConfirm,
  disabled,
  className,
}: {
  children: ReactNode;
  confirmLabel: string;
  question: ReactNode;
  onConfirm: () => void;
  disabled?: boolean;
  className?: string;
}) {
  const [asking, setAsking] = useState(false);
  if (!asking) {
    return (
      <button
        type="button"
        className={className}
        disabled={disabled}
        onClick={() => {
          setAsking(true);
        }}
      >
        {children}
      </button>
    );
  }
  return (
    <span className="inline-confirm" role="group" aria-label="Confirm">
      <span>{question}</span>
      <button
        type="button"
        className="btn-danger btn-small"
        disabled={disabled}
        onClick={() => {
          setAsking(false);
          onConfirm();
        }}
      >
        {confirmLabel}
      </button>
      <button
        type="button"
        className="btn-small"
        onClick={() => {
          setAsking(false);
        }}
      >
        Keep going
      </button>
    </span>
  );
}

export function Loading({ what }: { what: string }) {
  return (
    <div className="empty" role="status">
      Loading {what}…
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
