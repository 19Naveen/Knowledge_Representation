type EmptyStateProps = {
  title: string;
  description: string;
  actionLabel?: string;
};

export function EmptyState({ title, description, actionLabel }: EmptyStateProps) {
  return (
    <section className="rounded-xl border border-dashed bg-surface p-8 text-center shadow-sm">
      <h3 className="text-base font-semibold text-text">{title}</h3>
      <p className="mx-auto mt-2 max-w-sm text-sm text-muted">{description}</p>
      {actionLabel ? (
        <button
          type="button"
          className="mt-5 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white transition hover:opacity-90 shadow-sm"
        >
          {actionLabel}
        </button>
      ) : null}
    </section>
  );
}
