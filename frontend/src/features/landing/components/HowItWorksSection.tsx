import { steps } from "../landingContent";

export function HowItWorksSection() {
  return (
    <section id="how-it-works" className="py-24 px-6 bg-surface/30">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16 space-y-4">
          <p className="text-xs font-black uppercase tracking-widest text-primary">How It Works</p>
          <h2 className="text-4xl sm:text-5xl font-black tracking-tight">
            From raw data to predictions <br className="hidden sm:block" />in four steps
          </h2>
        </div>

        {/* Desktop layout */}
        <div className="hidden md:block">
          {/* Circle row with single line behind all circles */}
          <div className="relative grid grid-cols-4">
            {/* Full-width line behind circles */}
            <div className="absolute top-1/2 -translate-y-1/2 left-[12.5%] right-[12.5%] h-px bg-primary/20" />
            {steps.map((step) => (
              <div key={step.number} className="flex justify-center z-100">
                <div className="relative z-10 size-16 rounded-2xl bg-gray-200 border border-primary/20 flex items-center justify-center shadow-lg shadow-primary/10">
                  <span className="text-xl font-black text-primary">{step.number}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Description row */}
          <div className="grid grid-cols-4 gap-4 mt-6">
            {steps.map((step) => (
              <div key={step.number} className="text-center px-3">
                <h3 className="font-black text-sm tracking-tight">{step.title}</h3>
                <p className="text-sm text-text-secondary mt-2 leading-relaxed">{step.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Mobile layout */}
        <div className="md:hidden space-y-0">
          {steps.map((step, idx) => (
            <div key={step.number} className="flex flex-col items-center">
              <div className="flex items-center gap-4 w-full max-w-sm">
                <div className="flex flex-col items-center shrink-0">
                  {/* Top connector */}
                  {idx > 0 && <div className="w-px h-6 bg-primary/20" />}
                  <div className="size-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center shadow-lg shadow-primary/10">
                    <span className="text-lg font-black text-primary">{step.number}</span>
                  </div>
                  {/* Bottom connector */}
                  {idx < steps.length - 1 && <div className="w-px h-6 bg-primary/20" />}
                </div>
                <div className="py-4">
                  <h3 className="font-black text-base tracking-tight">{step.title}</h3>
                  <p className="text-sm text-text-secondary mt-1 leading-relaxed">{step.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
