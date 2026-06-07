import { LandingNavbar } from "./components/LandingNavbar";
import { HeroSection } from "./components/HeroSection";
import { FeaturesSection } from "./components/FeaturesSection";
import { HowItWorksSection } from "./components/HowItWorksSection";
import { CtaBanner } from "./components/CtaBanner";
import { LandingFooter } from "./components/LandingFooter";

export function LandingPage() {
  return (
    <div className="min-h-screen bg-bg text-text">
      <LandingNavbar />
      <main>
        <HeroSection />
        <FeaturesSection />
        <HowItWorksSection />
        <CtaBanner />
      </main>
      <LandingFooter />
    </div>
  );
}
