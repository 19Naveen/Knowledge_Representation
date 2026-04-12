type EmptyStateProps = {
  title: string;
  description: string;
  actionLabel?: string;
};

export function EmptyState({ title, description, actionLabel }: EmptyStateProps) {
  return (
    <section className="rounded-lg border border-dashed border-border bg-surface p-8 text-center">
      <h3 className="text-sm font-semibold text">{title}</h3>
      <p className="mx-auto mt-1.5 max-w-sm text-[13px] text-text-secondary">{description}</p>
      {actionLabel ? (
        <button
          type="button"
          className="btn btn-primary mt-4 text-sm"
        >
          {actionLabel}
        </button>
      ) : null}
    </section>
  );
}
