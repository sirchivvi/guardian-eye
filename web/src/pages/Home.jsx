import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Camera,
  Mic,
  ShieldCheck,
  Sparkles,
  ArrowRight,
  CheckCircle,
  Trees,
  Compass,
  AlertCircle,
  Volume2,
  Eye,
} from 'lucide-react';
import { AnimatedCounter } from '../components/AnimatedCounter';
import { Card } from '../components/Card';
import { RiskBadge } from '../components/RiskBadge';
import {
  FoxSvg,
  ElephantSvg,
  OwlSvg,
  DeerSvg,
  CameraTrapSvg,
  AcousticMicSvg,
  RangerShieldSvg,
  SproutSvg,
} from '../components/Animals';

export const Home = () => {
  useEffect(() => {
    document.title = 'GuardianEye — Helping Protect Wildlife, One Signal at a Time';
  }, []);

  return (
    <div className="space-y-20 pb-16">
      {/* Hero Section */}
      <section className="relative overflow-hidden pt-8 sm:pt-14 pb-12">
        {/* Soft background ambient gradient shapes */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-7xl h-96 bg-gradient-to-b from-[#F2F7F4] via-transparent to-transparent -z-10 rounded-full blur-3xl opacity-70 pointer-events-none" />

        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            {/* Left Column: Text & CTAs */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="lg:col-span-7 space-y-6 text-center lg:text-left"
            >
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#E2ECE5] border border-[#C4D9CC] text-[#385542] text-xs sm:text-sm font-bold shadow-sm">
                <SproutSvg className="w-4 h-4" />
                <span>Wildlife Conservation AI • Open Academic Proof-of-Concept</span>
              </div>

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-[#2D3748] tracking-tight font-display leading-[1.15]">
                Helping protect wildlife,{' '}
                <span className="text-[#5B8266] inline-block underline decoration-[#FDE68A] decoration-wavy decoration-2">
                  one signal
                </span>{' '}
                at a time.
              </h1>

              <p className="text-lg sm:text-xl text-[#4A5568] leading-relaxed max-w-2xl mx-auto lg:mx-0 font-normal">
                GuardianEye combines camera-trap vision with acoustic soundscapes to give park
                rangers a calm, reliable risk level (<span className="text-[#38A169] font-bold">LOW</span>,{' '}
                <span className="text-[#DD6B20] font-bold">MEDIUM</span>, or{' '}
                <span className="text-[#E05344] font-bold">HIGH</span>)—spotting potential poaching
                without panicking false alarms.
              </p>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4 pt-2">
                <Link
                  to="/try"
                  className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-7 py-4 rounded-full bg-gradient-to-r from-[#D96B43] to-[#C2542B] text-white font-extrabold text-base shadow-[0_6px_20px_rgba(217,107,67,0.35)] hover:shadow-[0_8px_25px_rgba(217,107,67,0.45)] hover:-translate-y-0.5 active:translate-y-0 transition-all"
                >
                  <Sparkles className="w-5 h-5 text-[#FDE68A]" />
                  <span>Try the Risk Checker</span>
                  <ArrowRight className="w-4 h-4 ml-1" />
                </Link>

                <Link
                  to="/results"
                  className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-4 rounded-full bg-white hover:bg-[#F2F7F4] text-[#385542] border border-[#C4D9CC] font-bold text-base shadow-sm hover:-translate-y-0.5 active:translate-y-0 transition-all"
                >
                  <Compass className="w-5 h-5 text-[#5B8266]" />
                  <span>Explore Benchmarks</span>
                </Link>
              </div>

              {/* Trust Badges */}
              <div className="pt-4 flex flex-wrap items-center justify-center lg:justify-start gap-5 text-xs text-[#718096] font-semibold">
                <span className="flex items-center gap-1.5">
                  <CheckCircle className="w-4 h-4 text-[#5B8266]" /> No black-box guesses
                </span>
                <span className="flex items-center gap-1.5">
                  <CheckCircle className="w-4 h-4 text-[#5B8266]" /> Strictly calibrated confidence
                </span>
                <span className="flex items-center gap-1.5">
                  <CheckCircle className="w-4 h-4 text-[#5B8266]" /> Respects ranger privacy
                </span>
              </div>
            </motion.div>

            {/* Right Column: Hero Interactive Graphic */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.7, delay: 0.15 }}
              className="lg:col-span-5 relative"
            >
              <div className="relative mx-auto max-w-md bg-gradient-to-b from-white to-[#FFFDF9] rounded-[2.5rem] p-7 sm:p-8 border-2 border-amber-100 shadow-[0_20px_50px_rgba(91,130,102,0.12)] space-y-6">
                {/* Decorative animal peek */}
                <div className="absolute -top-6 -right-4 bg-white p-2.5 rounded-2xl border border-amber-200 shadow-md transform rotate-6">
                  <DeerSvg className="w-9 h-9" />
                </div>

                <div className="flex items-center justify-between border-b border-amber-100 pb-4">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-[#5B8266]" />
                    <span className="text-xs font-extrabold uppercase tracking-wider text-[#718096]">
                      Active Forest Station #24
                    </span>
                  </div>
                  <RiskBadge level="LOW" size="sm" pulse={false} />
                </div>

                {/* Sensed Signals Mock */}
                <div className="space-y-3.5">
                  <div className="flex items-center gap-3.5 p-3.5 rounded-2xl bg-[#F2F7F4] border border-[#C4D9CC]/80">
                    <div className="w-11 h-11 rounded-xl bg-white flex items-center justify-center shadow-xs">
                      <CameraTrapSvg className="w-7 h-7" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-bold text-[#5B8266] uppercase">Camera Trap View</p>
                      <p className="text-sm font-extrabold text-[#2D3748] truncate">
                        Wildlife Detected (Elephant herd)
                      </p>
                    </div>
                    <span className="text-xs font-bold text-[#385542] bg-white px-2.5 py-1 rounded-full">
                      P(threat) = 0.04
                    </span>
                  </div>

                  <div className="flex items-center gap-3.5 p-3.5 rounded-2xl bg-[#FFF9F0] border border-amber-200/80">
                    <div className="w-11 h-11 rounded-xl bg-white flex items-center justify-center shadow-xs">
                      <AcousticMicSvg className="w-7 h-7" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-bold text-[#D96B43] uppercase">Acoustic Array</p>
                      <p className="text-sm font-extrabold text-[#2D3748] truncate">
                        Ambient Forest (Cicadas & wind)
                      </p>
                    </div>
                    <span className="text-xs font-bold text-[#7B341E] bg-white px-2.5 py-1 rounded-full">
                      P(threat) = 0.02
                    </span>
                  </div>
                </div>

                {/* Fusion Result Card */}
                <div className="p-4 rounded-2xl bg-gradient-to-r from-[#EAF7EE] to-[#E2ECE5] border border-[#A3E6B7] space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-extrabold text-[#22543D] uppercase tracking-wide">
                      Fused Assessment
                    </span>
                    <span className="text-xs font-extrabold text-[#22543D]">Score: 0.03 / 1.00</span>
                  </div>
                  <p className="text-xs text-[#22543D] leading-relaxed">
                    Visual indicates grazing wildlife; acoustic stream shows natural background sound.
                    No intervention required. Rangers keep watch safely.
                  </p>
                </div>

                {/* Tiny cute footer on card */}
                <div className="flex items-center justify-between text-xs text-[#718096] pt-1">
                  <span className="flex items-center gap-1.5">
                    <FoxSvg className="w-4 h-4 inline" /> Protected Sanctuary
                  </span>
                  <span>Latency: ~18ms</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto mb-14 space-y-3">
          <span className="inline-block px-3.5 py-1 rounded-full bg-[#FEF3C7] text-[#92400E] text-xs font-extrabold uppercase tracking-wider">
            Three Simple Steps
          </span>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-[#2D3748] font-display">
            How GuardianEye Works
          </h2>
          <p className="text-[#4A5568] text-base sm:text-lg">
            Neither a camera nor a microphone is enough alone. When brought together, they provide
            clear situational awareness for conservation teams.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Step 1 */}
          <Card className="relative overflow-hidden group">
            <div className="w-14 h-14 rounded-2xl bg-[#E2ECE5] flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <CameraTrapSvg className="w-9 h-9" />
            </div>
            <div className="inline-block px-2.5 py-0.5 rounded-full bg-amber-100/70 text-[#78350F] text-xs font-black uppercase mb-3">
              Step 1
            </div>
            <h3 className="text-xl font-extrabold text-[#2D3748] mb-2 font-display">
              1. Camera Sees
            </h3>
            <p className="text-sm text-[#4A5568] leading-relaxed">
              Camera traps and thermal drones detect motion along trails, identifying whether
              movement is a human, vehicle, animal, or an empty tree branch swayed by the wind.
            </p>
            <div className="mt-5 pt-4 border-t border-amber-100 flex items-center text-xs text-[#5B8266] font-bold">
              <span>Classes: Human, Vehicle, Animal, Empty</span>
            </div>
          </Card>

          {/* Step 2 */}
          <Card className="relative overflow-hidden group">
            <div className="w-14 h-14 rounded-2xl bg-[#F9E5DE] flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <AcousticMicSvg className="w-9 h-9" />
            </div>
            <div className="inline-block px-2.5 py-0.5 rounded-full bg-amber-100/70 text-[#78350F] text-xs font-black uppercase mb-3">
              Step 2
            </div>
            <h3 className="text-xl font-extrabold text-[#2D3748] mb-2 font-display">
              2. Microphone Listens
            </h3>
            <p className="text-sm text-[#4A5568] leading-relaxed">
              Acoustic sensors continuously listen for the distinct acoustic signature of gunshots,
              trained to differentiate firearms from falling logs, thunderclaps, and tropical downpours.
            </p>
            <div className="mt-5 pt-4 border-t border-amber-100 flex items-center text-xs text-[#D96B43] font-bold">
              <span>Classes: Gunshot vs. Ambient Forest</span>
            </div>
          </Card>

          {/* Step 3 */}
          <Card className="relative overflow-hidden group">
            <div className="w-14 h-14 rounded-2xl bg-[#EAF7EE] flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <RangerShieldSvg className="w-9 h-9" />
            </div>
            <div className="inline-block px-2.5 py-0.5 rounded-full bg-amber-100/70 text-[#78350F] text-xs font-black uppercase mb-3">
              Step 3
            </div>
            <h3 className="text-xl font-extrabold text-[#2D3748] mb-2 font-display">
              3. We Assess Risk
            </h3>
            <p className="text-sm text-[#4A5568] leading-relaxed">
              Our late-fusion decision head evaluates both streams. If a camera sees a vehicle AND
              a gunshot is heard, risk escalates to <span className="text-[#E05344] font-bold">HIGH</span>.
              If just an animal passes, risk stays <span className="text-[#38A169] font-bold">LOW</span>.
            </p>
            <div className="mt-5 pt-4 border-t border-amber-100 flex items-center text-xs text-[#38A169] font-bold">
              <span>Risk tiers: LOW • MEDIUM • HIGH</span>
            </div>
          </Card>
        </div>
      </section>

      {/* Stats Counter Section */}
      <section className="bg-gradient-to-r from-[#FFFDF9] via-[#FAF2E8] to-[#FFFDF9] py-14 border-y border-amber-200/50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 text-center">
            <div className="space-y-1">
              <div className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-[#5B8266] font-display">
                <AnimatedCounter target={94.2} decimals={1} suffix="%" />
              </div>
              <p className="text-xs sm:text-sm font-extrabold text-[#2D3748] uppercase tracking-wider">
                Gunshot Recall
              </p>
              <p className="text-xs text-[#718096]">Acoustic bioacoustic branch</p>
            </div>

            <div className="space-y-1">
              <div className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-[#D96B43] font-display">
                <AnimatedCounter target={89.8} decimals={1} suffix="%" />
              </div>
              <p className="text-xs sm:text-sm font-extrabold text-[#2D3748] uppercase tracking-wider">
                Vision Test Accuracy
              </p>
              <p className="text-xs text-[#718096]">4-class thermal & camera trap</p>
            </div>

            <div className="space-y-1">
              <div className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-[#F59E0B] font-display">
                <AnimatedCounter target={15000} suffix="+" />
              </div>
              <p className="text-xs sm:text-sm font-extrabold text-[#2D3748] uppercase tracking-wider">
                Synthetic Pairs
              </p>
              <p className="text-xs text-[#718096]">Rigorously isolated across seeds</p>
            </div>

            <div className="space-y-1">
              <div className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-[#76A284] font-display">
                <AnimatedCounter target={100} suffix="%" />
              </div>
              <p className="text-xs sm:text-sm font-extrabold text-[#2D3748] uppercase tracking-wider">
                Honest Validation
              </p>
              <p className="text-xs text-[#718096]">Never claims unverified field data</p>
            </div>
          </div>
        </div>
      </section>

      {/* Why Both Signals Matter Interactive Comparison */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto mb-12 space-y-2">
          <h2 className="text-3xl sm:text-4xl font-extrabold text-[#2D3748] font-display">
            Why Single Sensors Fail in the Wild
          </h2>
          <p className="text-[#4A5568]">
            Dense rainforests, monsoons, and night darkness deceive unimodal sensors. Combining
            sight and sound resolves the ambiguity.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Case 1: Audio false alarm saved by camera */}
          <Card className="space-y-4 border-l-4 border-l-[#5B8266]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-extrabold uppercase px-3 py-1 rounded-full bg-[#E2ECE5] text-[#385542]">
                Scenario A: Heavy Thunderstorm
              </span>
              <RiskBadge level="LOW" size="sm" pulse={false} />
            </div>
            <h4 className="text-lg font-extrabold text-[#2D3748] font-display">
              Acoustic Boom + Empty Forest
            </h4>
            <p className="text-sm text-[#4A5568] leading-relaxed">
              A lightning strike rattles the valley. An audio-only detector might mistakenly flag a
              high-energy impulse as a rifle shot. GuardianEye cross-references the camera view: no
              human or vehicle tracks are visible. The risk remains sensibly classified as{' '}
              <span className="font-bold text-[#38A169]">LOW</span>, sparing rangers an exhausting
              midnight trek.
            </p>
            <div className="p-3 rounded-2xl bg-[#FFF9F0] text-xs text-[#718096] flex items-center gap-2">
              <OwlSvg className="w-5 h-5 flex-shrink-0" />
              <span>Result: Protects rangers against alert fatigue and false alarms.</span>
            </div>
          </Card>

          {/* Case 2: Visual blind spot caught by sound */}
          <Card className="space-y-4 border-l-4 border-l-[#D96B43]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-extrabold uppercase px-3 py-1 rounded-full bg-[#F9E5DE] text-[#9C3F1D]">
                Scenario B: Night Camouflage
              </span>
              <RiskBadge level="MEDIUM" size="sm" pulse={false} />
            </div>
            <h4 className="text-lg font-extrabold text-[#2D3748] font-display">
              Obscured Fog + Distant Gunshot
            </h4>
            <p className="text-sm text-[#4A5568] leading-relaxed">
              At 2 AM, dense jungle foliage blocks the camera trap's infrared flash. However, the
              acoustic microphone picks up a sharp gunshot signature echoing from the riverbank.
              Even with vision uncertain, the system elevates risk to{' '}
              <span className="font-bold text-[#DD6B20]">MEDIUM</span> so headquarters can monitor
              the perimeter without delay.
            </p>
            <div className="p-3 rounded-2xl bg-[#FFF9F0] text-xs text-[#718096] flex items-center gap-2">
              <ElephantSvg className="w-5 h-5 flex-shrink-0" />
              <span>Result: Critical acoustic incidents aren't missed when visual field of view is blocked.</span>
            </div>
          </Card>
        </div>
      </section>

      {/* Ranger Story & Quote Section */}
      <section className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="rounded-[2.5rem] bg-gradient-to-br from-[#E2ECE5] via-[#F2F7F4] to-[#FAF2E8] p-8 sm:p-12 border border-[#C4D9CC] shadow-sm relative overflow-hidden">
          <div className="relative z-10 space-y-4 text-center sm:text-left sm:flex items-center gap-8">
            <div className="w-24 h-24 rounded-full bg-white mx-auto sm:mx-0 p-3 shadow-md flex items-center justify-center flex-shrink-0">
              <ElephantSvg className="w-16 h-16" />
            </div>
            <div className="space-y-2">
              <p className="text-base sm:text-lg italic text-[#2D3748] leading-relaxed">
                “In conservation, an alarm system that cries wolf causes rangers to ignore real
                emergencies. A system that quietly confirms what it sees and hears gives us the
                confidence to protect rhinos and elephants safely.”
              </p>
              <div className="pt-2">
                <p className="text-sm font-extrabold text-[#385542]">
                  Simulated Ranger Field Coordinator
                </p>
                <p className="text-xs text-[#718096]">
                  Belize & Vietnam Biosphere Reserve Case Study
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Big Friendly CTA */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-gradient-to-r from-[#5B8266] to-[#4A6E55] rounded-[2.5rem] p-10 sm:p-14 text-white text-center space-y-6 shadow-[0_15px_40px_rgba(91,130,102,0.35)] relative overflow-hidden">
          <div className="max-w-xl mx-auto space-y-3 relative z-10">
            <h2 className="text-3xl sm:text-4xl font-extrabold font-display">
              Ready to test the risk logic?
            </h2>
            <p className="text-amber-100 text-base sm:text-lg">
              Pick what the camera sees and what the microphone hears in our interactive playground
              to watch the decision head calculate risk in real time.
            </p>
            <div className="pt-4">
              <Link
                to="/try"
                className="inline-flex items-center gap-2.5 px-8 py-4 rounded-full bg-white text-[#385542] hover:bg-amber-50 font-extrabold text-base shadow-lg hover:scale-105 active:scale-100 transition-transform"
              >
                <Sparkles className="w-5 h-5 text-[#D96B43]" />
                <span>Launch Interactive Risk Checker</span>
              </Link>
            </div>
          </div>

          {/* Decorative animals in corners */}
          <div className="absolute -bottom-4 -left-4 opacity-25 pointer-events-none">
            <FoxSvg className="w-32 h-32" />
          </div>
          <div className="absolute -top-6 -right-6 opacity-25 pointer-events-none">
            <OwlSvg className="w-32 h-32" />
          </div>
        </div>
      </section>
    </div>
  );
};
