export function PageHeader({ title, subtitle, actions, breadcrumbs }: { title: string; subtitle?: string; actions?: React.ReactNode; breadcrumbs?: string[] }) {
  return (
    <header className="mb-6 flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
      <div>
        {breadcrumbs && (
          <nav className="mb-2 flex items-center gap-1.5 text-xs text-text-tertiary">
            {breadcrumbs.map((crumb, i) => (
              <span key={i}>{i > 0 && <span className="mx-1">/</span>}{crumb}</span>
            ))}
          </nav>
        )}
        <h1 className="text-2xl font-heading font-semibold tracking-tight text-text">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-text-secondary">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
}