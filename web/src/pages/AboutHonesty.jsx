import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  ShieldCheck,
  AlertCircle,
  Trees,
  Heart,
  Scale,
  Database,
  Cpu,
  Compass,
  FileCheck,
  CheckCircle2,
} from 'lucide-react';
import { Card } from '../components/Card';
import {
  FoxSvg,
  ElephantSvg,
  OwlSvg,
  DeerSvg,
  SproutSvg,
  RangerShieldSvg,
  CameraTrapSvg,
  AcousticMicSvg,
} from '../components/Animals';

export const AboutHonesty = () => {
  useEffect(() => {
    document.title = 'GuardianEye — About & Honesty Policy';
  }, []);

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto space-y-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#E2ECE5] text-[#385542] text-xs font-bold shadow-xs">
          <Scale className="w-3.5 h-3.5 text-[#5B8266]" />
          <span>Ethics, Transparency & Scientific Rigor</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold text-[#2D3748] font-display">
          About GuardianEye & Our Honesty Policy
        </h1>
        <p className="text-base sm:text-lg text-[#4A5568]">
          We believe artificial intelligence in conservation must be candid about what it can do,
          how it was evaluated, and where laboratory data meets real-world wild forests.
        </p>
      </div>

      {/* Primary Academic Disclosure Banner */}
      <div className="rounded-[2.5rem] bg-gradient-to-r from-[#FFF5EA] via-white to-[#FFF5EA] p-8 sm:p-10 border-2 border-[#FBD38D] shadow-[0_10px_35px_rgba(221,107,32,0.08)] space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-2xl bg-[#DD6B20] text-white flex items-center justify-center flex-shrink-0 shadow-sm">
            <RangerShieldSvg className="w-8 h-8" />
          </div>
          <div>
            <h2 className="text-xl sm:text-2xl font-extrabold text-[#7B341E] font-display">
              Academic Proof of Concept — Not an Operational Field Alarm
            </h2>
            <p className="text-xs sm:text-sm text-[#92400E] font-medium">
              GuardianEye is an open scientific exploration, not deployable tactical security software.
            </p>
          </div>
        </div>
        <p className="text-sm text-[#4A5568] leading-relaxed">
          GuardianEye was created to demonstrate how multi-sensor fusion (vision + bioacoustics) can
          theoretically enhance ranger situational awareness. It has{' '}
          <strong className="text-[#2D3748]">not been tested in an active live-patrol setting</strong>{' '}
          with connected edge hardware. Park rangers should never rely exclusively on these models
          for life-safety decisions without extensive local ground calibration.
        </p>
      </div>

      {/* Key Principle 1: Synthetic Pairing Transparency */}
      <section className="space-y-4">
        <div className="flex items-center gap-2 text-xl font-extrabold text-[#2D3748] font-display">
          <Database className="w-6 h-6 text-[#5B8266]" />
          <h3>Synthetic Image–Audio Pairings & Split Integrity</h3>
        </div>

        <Card className="space-y-4 border-l-4 border-l-[#5B8266]">
          <p className="text-sm text-[#4A5568] leading-relaxed">
            In the wild, synchronized multimodal datasets—where a camera trap photograph and an
            acoustic sound recording capture the exact same event at the exact millisecond—are
            virtually nonexistent in public conservation archives.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
            <div className="p-4 rounded-2xl bg-[#F2F7F4] border border-[#C4D9CC] space-y-1.5">
              <span className="text-xs font-bold uppercase text-[#385542] flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-[#5B8266]" /> Strict Split Isolation
              </span>
              <p className="text-xs text-[#4A5568] leading-relaxed">
                Images and audio clips are paired strictly within their designated train, validation,
                or test partition. A training audio clip is never paired with a test image.
              </p>
            </div>

            <div className="p-4 rounded-2xl bg-[#FFF9F0] border border-amber-200 space-y-1.5">
              <span className="text-xs font-bold uppercase text-[#D96B43] flex items-center gap-1.5">
                <AlertCircle className="w-4 h-4 text-[#D96B43]" /> Honest Ground Truth Rule
              </span>
              <p className="text-xs text-[#4A5568] leading-relaxed">
                We never treat heuristic synthetic pairings as independently observed threat ground
                truth. Benchmarks measure algorithmic consistency, not simulated field captures.
              </p>
            </div>
          </div>
        </Card>
      </section>

      {/* Key Principle 2: Datasets & Limitations */}
      <section className="space-y-4">
        <div className="flex items-center gap-2 text-xl font-extrabold text-[#2D3748] font-display">
          <CameraTrapSvg className="w-7 h-7" />
          <h3>Datasets Used & Their Inherent Limitations</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-extrabold uppercase text-[#5B8266] bg-[#E2ECE5] px-2.5 py-1 rounded-full">
                Visual Modality
              </span>
              <span className="text-xs text-[#718096]">RGB + Thermal</span>
            </div>
            <h4 className="text-base font-extrabold text-[#2D3748] font-display">
              BIRDSAI (Thermal UAV) & Caltech (CCT-20)
            </h4>
            <p className="text-xs text-[#4A5568] leading-relaxed">
              <strong>BIRDSAI</strong> provides aerial thermal infrared footage taken from unmanned
              aerial vehicles (drones), while <strong>CCT-20</strong> consists of ground-level optical
              camera-trap photographs. While complementary, an algorithm trained on drone thermal
              silhouettes behaves differently from trail cameras hidden behind dense brush.
            </p>
          </Card>

          <Card className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-extrabold uppercase text-[#D96B43] bg-[#F9E5DE] px-2.5 py-1 rounded-full">
                Acoustic Modality
              </span>
              <span className="text-xs text-[#718096]">Bioacoustic Arrays</span>
            </div>
            <h4 className="text-base font-extrabold text-[#2D3748] font-display">
              Belize Jungle & Vietnam Chu Mom Ray
            </h4>
            <p className="text-xs text-[#4A5568] leading-relaxed">
              Acoustic models were trained on rainforest gunshot recordings from Belize and Vietnam,
              benchmarked against environmental wind and rain from ESC-50. Acoustic reverberation in
              steep mountainous terrain differs substantially from flat savanna or wetlands.
            </p>
          </Card>
        </div>
      </section>

      {/* Key Principle 3: Ethics & Ranger Welfare */}
      <section className="space-y-4">
        <div className="flex items-center gap-2 text-xl font-extrabold text-[#2D3748] font-display">
          <Heart className="w-6 h-6 text-[#E07A5F]" />
          <h3>Conservation Ethics & Ranger Welfare</h3>
        </div>

        <Card className="space-y-4 bg-gradient-to-r from-white via-[#FFFDF9] to-white">
          <div className="space-y-3 text-sm text-[#4A5568] leading-relaxed">
            <p>
              Wildlife rangers work in some of the most dangerous conditions on earth, patrolling
              vast remote sanctuaries against armed poachers while managing encounters with dangerous
              wild animals.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
              <div className="p-4 rounded-2xl bg-white border border-amber-100 shadow-xs space-y-1">
                <span className="text-base">🛡️</span>
                <p className="text-xs font-bold text-[#2D3748]">Preventing Alert Fatigue</p>
                <p className="text-[11px] text-[#718096]">
                  A noisy detector that alarms on thunder or monkeys causes rangers to tune out alerts.
                </p>
              </div>
              <div className="p-4 rounded-2xl bg-white border border-amber-100 shadow-xs space-y-1">
                <span className="text-base">🤝</span>
                <p className="text-xs font-bold text-[#2D3748]">Community Respect</p>
                <p className="text-[11px] text-[#718096]">
                  Cameras in natural corridors must never be turned into surveillance tools against local populations.
                </p>
              </div>
              <div className="p-4 rounded-2xl bg-white border border-amber-100 shadow-xs space-y-1">
                <span className="text-base">🌿</span>
                <p className="text-xs font-bold text-[#2D3748]">Open Science</p>
                <p className="text-[11px] text-[#718096]">
                  All code, architectures, and evaluation harnesses are transparent and reproducible.
                </p>
              </div>
            </div>
          </div>
        </Card>
      </section>

      {/* Friendly animal closing banner */}
      <div className="p-8 rounded-[2.5rem] bg-[#E2ECE5] border border-[#C4D9CC] flex flex-col sm:flex-row items-center gap-6 text-center sm:text-left">
        <div className="w-16 h-16 rounded-full bg-white p-2.5 flex items-center justify-center flex-shrink-0 shadow-sm">
          <OwlSvg className="w-10 h-10" />
        </div>
        <div className="space-y-1">
          <h4 className="text-base font-extrabold text-[#385542] font-display">
            A gentle reminder from the GuardianEye team
          </h4>
          <p className="text-xs sm:text-sm text-[#4A5568] leading-relaxed">
            Conservation AI is not magic. It is humble, patient pair programming between machine
            learning researchers and park rangers on the ground. Thank you for caring for our
            planet's living heritage.
          </p>
        </div>
      </div>
    </div>
  );
};
