import { features } from "../landingContent";

export function FeaturesSection() {
  return (
    <section id="features" className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16 space-y-4">
          <p className="text-xs font-black uppercase tracking-widest text-primary">Platform Capabilities</p>
          <h2 className="text-4xl sm:text-5xl font-black tracking-tight">
            Everything your data team needs
          </h2>
          <p className="text-text-secondary text-lg max-w-2xl mx-auto">
            From raw ingestion to production ML — one platform, zero compromises.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="card p-6 group hover:border-primary/30 hover:shadow-2xl hover:shadow-primary/5 transition-all duration-300 space-y-4"
            >
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center bg-surface-2 border border-border group-hover:bg-primary/10 group-hover:border-primary/20 transition-all ${feature.accent}`}>
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={feature.icon} />
                </svg>
              </div>
              <div>
                <h3 className="font-bold text-base tracking-tight mb-1.5">{feature.title}</h3>
                <p className="text-sm text-text-secondary leading-relaxed">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
