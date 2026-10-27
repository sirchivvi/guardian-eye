import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  Sparkles,
  Camera,
  Volume2,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  Info,
  RefreshCw,
  HelpCircle,
  ImageIcon,
} from 'lucide-react';
import { Card } from '../components/Card';
import { RiskBadge } from '../components/RiskBadge';
import {
  FoxSvg,
  ElephantSvg,
  OwlSvg,
  DeerSvg,
  CameraTrapSvg,
  AcousticMicSvg,
} from '../components/Animals';

export const TryIt = () => {
  useEffect(() => {
    document.title = 'GuardianEye — Interactive Risk Checker';
  }, []);

  // State
  const [cameraSelection, setCameraSelection] = useState('human');
  const [audioSelection, setAudioSelection] = useState('gunshot');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState('');

  // Big friendly selection options
  const cameraOptions = [
    { id: 'human', label: 'Human', icon: '🚶', desc: 'Poacher or unauthorized entrant' },
    { id: 'vehicle', label: 'Vehicle', icon: '🚙', desc: 'Motorbike, 4x4, or truck' },
    { id: 'animal', label: 'Animal', icon: '🐘', desc: 'Elephant, rhino, or deer' },
    { id: 'empty', label: 'Empty Forest', icon: '🍃', desc: 'Trees, vegetation, swaying brush' },
  ];

  const audioOptions = [
    { id: 'gunshot', label: 'Gunshot', icon: '💥', desc: 'Acoustic weapon blast signature' },
    { id: 'background', label: 'Forest Background', icon: '🌿', desc: 'Wind, rain, insect chorus, thunder' },
  ];

  // Presets
  const presets = [
    {
      title: '🚨 Poaching Threat',
      cam: 'human',
      aud: 'gunshot',
      badge: 'HIGH',
    },
    {
      title: '🚗 Vehicle Trespass',
      cam: 'vehicle',
      aud: 'background',
      badge: 'MEDIUM',
    },
    {
      title: '🦌 Wildlife Grazing',
      cam: 'animal',
      aud: 'background',
      badge: 'LOW',
    },
    {
      title: '🔊 Distant Sound Echo',
      cam: 'empty',
      aud: 'gunshot',
      badge: 'MEDIUM',
    },
    {
      title: '🌲 Quiet Forest Trail',
      cam: 'empty',
      aud: 'background',
      badge: 'LOW',
    },
  ];

  // Decision Logic per exact prompt rules:
  // - HIGH: human OR vehicle + gunshot
  // - MEDIUM: animal + gunshot, OR human OR vehicle + background, OR empty + gunshot
  // - LOW: animal OR empty + background
  const calculateRisk = (cam, aud) => {
    if ((cam === 'human' || cam === 'vehicle') && aud === 'gunshot') {
      return {
        level: 'HIGH',
        headline: 'Critical Threat Alert: Authorized Intervention Required',
        explanation:
          'Camera detected armed human or vehicle movement in synchrony with verified gunshot acoustic acoustics. High probability of active poaching.',
        action: 'Dispatch armed tactical ranger squad to coordinates immediately. Exercise extreme caution.',
        simulatedScore: 0.94,
        camProb: cam === 'human' ? 0.92 : 0.88,
        audProb: 0.96,
        colorBg: 'bg-[#FFF0EE]',
        colorBorder: 'border-[#FEB2B2]',
        colorText: 'text-[#742A2A]',
      };
    }

    if (cam === 'animal' && aud === 'gunshot') {
      return {
        level: 'MEDIUM',
        headline: 'Acoustic Threat Near Wildlife Field of View',
        explanation:
          'Gunshot acoustic signature detected, but camera view shows wildlife. The firearm may have discharged nearby, just outside camera angle.',
        action: 'Notify patrol coordinator. Check adjacent acoustic sensors and thermal drone feeds.',
        simulatedScore: 0.62,
        camProb: 0.05,
        audProb: 0.95,
        colorBg: 'bg-[#FFF5EA]',
        colorBorder: 'border-[#FBD38D]',
        colorText: 'text-[#7B341E]',
      };
    }

    if ((cam === 'human' || cam === 'vehicle') && aud === 'background') {
      return {
        level: 'MEDIUM',
        headline: 'Visual Trespass Detected (No Weapon Fired)',
        explanation:
          `Visual capture shows ${cam === 'human' ? 'human entry' : 'vehicle passage'}, but microphone records only ambient forest audio without gunfire.`,
        action: 'Dispatch boundary verification team. Monitor track checkpoint for contraband or snares.',
        simulatedScore: 0.58,
        camProb: cam === 'human' ? 0.89 : 0.86,
        audProb: 0.04,
        colorBg: 'bg-[#FFF5EA]',
        colorBorder: 'border-[#FBD38D]',
        colorText: 'text-[#7B341E]',
      };
    }

    if (cam === 'empty' && aud === 'gunshot') {
      return {
        level: 'MEDIUM',
        headline: 'Acoustic Weapon Anomaly on Camera Blindspot',
        explanation:
          'Microphone acoustic sensor triggered on firearm report, but camera trap view has no visible subject. Sound may originate beyond camera line-of-sight.',
        action: 'Triangulate report with nearby acoustic array stations; deploy drone perimeter sweep.',
        simulatedScore: 0.54,
        camProb: 0.02,
        audProb: 0.95,
        colorBg: 'bg-[#FFF5EA]',
        colorBorder: 'border-[#FBD38D]',
        colorText: 'text-[#7B341E]',
      };
    }

    // Default LOW: animal or empty + background
    return {
      level: 'LOW',
      headline: 'Normal Ecosystem Activity — Safe Sanctuary',
      explanation:
        `Forest sensors record ${cam === 'animal' ? 'grazing wildlife' : 'an undisturbed natural clearing'} accompanied by natural ambient sound.`,
      action: 'No intervention needed. Log record into wildlife population monitoring archives.',
      simulatedScore: 0.05,
      camProb: cam === 'animal' ? 0.04 : 0.01,
      audProb: 0.02,
      colorBg: 'bg-[#EAF7EE]',
      colorBorder: 'border-[#A3E6B7]',
      colorText: 'text-[#22543D]',
    };
  };

  const riskResult = calculateRisk(cameraSelection, audioSelection);

  // Image Upload handler
  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFileName(file.name);
      const reader = new FileReader();
      reader.onload = (event) => {
        setUploadedImage(event.target?.result);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto space-y-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#E2ECE5] text-[#385542] text-xs font-bold shadow-xs">
          <Sparkles className="w-3.5 h-3.5 text-[#5B8266]" />
          <span>Interactive Risk Sandbox</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold text-[#2D3748] font-display">
          GuardianEye Risk Checker
        </h1>
        <p className="text-base sm:text-lg text-[#4A5568]">
          Select what the camera trap sees and what the bioacoustic array hears to observe how
          GuardianEye's fusion head determines ranger alert levels.
        </p>
      </div>

      {/* Quick Presets Bar */}
      <div className="bg-white rounded-3xl p-5 border border-amber-100/90 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-[#718096]">
          <FoxSvg className="w-4 h-4 inline" />
          <span>Try a Quick Preset Scenario:</span>
        </div>
        <div className="flex flex-wrap gap-2.5">
          {presets.map((preset) => (
            <button
              key={preset.title}
              onClick={() => {
                setCameraSelection(preset.cam);
                setAudioSelection(preset.aud);
              }}
              className={`px-3.5 py-2 rounded-full text-xs sm:text-sm font-bold border transition-all flex items-center gap-2 ${
                cameraSelection === preset.cam && audioSelection === preset.aud
                  ? 'bg-[#5B8266] text-white border-[#5B8266] shadow-sm'
                  : 'bg-[#FFFDF9] hover:bg-amber-50 text-[#4A5568] border-amber-200/70'
              }`}
            >
              <span>{preset.title}</span>
              <span
                className={`text-[10px] px-2 py-0.5 rounded-full font-black ${
                  preset.badge === 'HIGH'
                    ? 'bg-[#E05344] text-white'
                    : preset.badge === 'MEDIUM'
                    ? 'bg-[#F59E0B] text-white'
                    : 'bg-[#38A169] text-white'
                }`}
              >
                {preset.badge}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Interactive Controls Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Sensed Modality Selectors (7 cols) */}
        <div className="lg:col-span-7 space-y-8">
          {/* 1. Camera Trap Selector */}
          <Card className="space-y-4 border-2 border-amber-100">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-10 h-10 rounded-2xl bg-[#E2ECE5] flex items-center justify-center">
                  <Camera className="w-5 h-5 text-[#5B8266]" />
                </div>
                <div>
                  <h3 className="text-lg font-extrabold text-[#2D3748] font-display">
                    1. What did the camera trap see?
                  </h3>
                  <p className="text-xs text-[#718096]">Optical or thermal movement signature</p>
                </div>
              </div>
              <span className="text-xs font-bold text-[#5B8266] bg-[#E2ECE5] px-2.5 py-1 rounded-full">
                Modality: Image
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {cameraOptions.map((opt) => {
                const isSelected = cameraSelection === opt.id;
                return (
                  <button
                    key={opt.id}
                    onClick={() => setCameraSelection(opt.id)}
                    className={`p-4 rounded-2xl border text-center transition-all flex flex-col items-center justify-center gap-2 ${
                      isSelected
                        ? 'bg-[#E2ECE5] border-[#5B8266] text-[#2D3748] shadow-sm ring-2 ring-[#5B8266]/30'
                        : 'bg-[#FFFDF9] hover:bg-amber-50/60 border-amber-200/70 text-[#4A5568]'
                    }`}
                  >
                    <span className="text-3xl">{opt.icon}</span>
                    <span className="text-sm font-extrabold block">{opt.label}</span>
                    <span className="text-[11px] text-[#718096] leading-tight line-clamp-2">
                      {opt.desc}
                    </span>
                  </button>
                );
              })}
            </div>
          </Card>

          {/* 2. Microphone Selector */}
          <Card className="space-y-4 border-2 border-amber-100">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-10 h-10 rounded-2xl bg-[#F9E5DE] flex items-center justify-center">
                  <Volume2 className="w-5 h-5 text-[#D96B43]" />
                </div>
                <div>
                  <h3 className="text-lg font-extrabold text-[#2D3748] font-display">
                    2. What did the microphone hear?
                  </h3>
                  <p className="text-xs text-[#718096]">Acoustic log-mel spectrogram analysis</p>
                </div>
              </div>
              <span className="text-xs font-bold text-[#D96B43] bg-[#F9E5DE] px-2.5 py-1 rounded-full">
                Modality: Audio
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {audioOptions.map((opt) => {
                const isSelected = audioSelection === opt.id;
                return (
                  <button
                    key={opt.id}
                    onClick={() => setAudioSelection(opt.id)}
                    className={`p-4 rounded-2xl border text-left transition-all flex items-center gap-4 ${
                      isSelected
                        ? 'bg-[#F9E5DE] border-[#D96B43] text-[#2D3748] shadow-sm ring-2 ring-[#D96B43]/30'
                        : 'bg-[#FFFDF9] hover:bg-amber-50/60 border-amber-200/70 text-[#4A5568]'
                    }`}
                  >
                    <span className="text-3xl p-2 rounded-xl bg-white shadow-xs">{opt.icon}</span>
                    <div>
                      <span className="text-base font-extrabold block">{opt.label}</span>
                      <span className="text-xs text-[#718096] block">{opt.desc}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </Card>

          {/* 3. Image Upload Preview Sandbox */}
          <Card className="space-y-4 border border-dashed border-amber-300 bg-[#FFFDF9]">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-amber-100 flex items-center justify-center">
                  <Upload className="w-4 h-4 text-[#78350F]" />
                </div>
                <div>
                  <h4 className="text-sm font-extrabold text-[#2D3748]">
                    Camera Trap Image Upload Preview
                  </h4>
                  <p className="text-xs text-[#718096]">
                    Preview an image file (.jpg, .png)
                  </p>
                </div>
              </div>
              <span className="text-[11px] font-bold text-[#D96B43] bg-[#F9E5DE] px-2.5 py-0.5 rounded-full">
                Phase 6 Sandbox
              </span>
            </div>

            {uploadedImage ? (
              <div className="space-y-3">
                <div className="relative rounded-2xl overflow-hidden border border-amber-200 max-h-56 bg-black/5 flex items-center justify-center">
                  <img
                    src={uploadedImage}
                    alt="Uploaded camera trap preview"
                    className="max-h-56 object-contain"
                  />
                  <button
                    onClick={() => {
                      setUploadedImage(null);
                      setUploadedFileName('');
                    }}
                    className="absolute top-2 right-2 bg-black/60 hover:bg-black/80 text-white text-xs px-2.5 py-1 rounded-full backdrop-blur-xs font-bold"
                  >
                    Remove
                  </button>
                </div>
                <div className="p-3 rounded-xl bg-[#FEF3C7] border border-[#FDE68A] text-xs text-[#92400E] flex items-start gap-2">
                  <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="font-bold">Loaded: {uploadedFileName}</span>. Note: Live visual
                    inference server will connect in Phase 6. For now, use the interactive selector buttons
                    above to test the risk logic!
                  </div>
                </div>
              </div>
            ) : (
              <label className="flex flex-col items-center justify-center p-6 rounded-2xl border-2 border-dashed border-amber-200 hover:border-[#5B8266] bg-amber-50/30 hover:bg-amber-50/60 cursor-pointer transition-colors group">
                <ImageIcon className="w-8 h-8 text-[#76A284] group-hover:scale-110 transition-transform mb-2" />
                <span className="text-xs font-extrabold text-[#2D3748]">
                  Click to browse or drop an animal / camera-trap photo
                </span>
                <span className="text-[11px] text-[#718096] mt-0.5">
                  Local client preview only (no image data uploaded to server)
                </span>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </label>
            )}
          </Card>
        </div>

        {/* Right Column: Dynamic Animated Risk Result (5 cols) */}
        <div className="lg:col-span-5 sticky top-28 space-y-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={`${cameraSelection}-${audioSelection}`}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
            >
              <div
                className={`rounded-[2.5rem] p-7 sm:p-8 border-2 ${riskResult.colorBorder} ${riskResult.colorBg} shadow-[0_15px_40px_rgba(0,0,0,0.06)] space-y-6`}
              >
                {/* Result Header */}
                <div className="flex items-center justify-between border-b border-black/5 pb-4">
                  <div className="space-y-0.5">
                    <span className="text-xs font-extrabold uppercase tracking-wider text-[#718096]">
                      Multimodal Evaluation
                    </span>
                    <h3 className="text-lg font-black text-[#2D3748] font-display">
                      Calculated Risk
                    </h3>
                  </div>
                  <RiskBadge level={riskResult.level} size="lg" />
                </div>

                {/* Plain-Language Explanation */}
                <div className="space-y-2">
                  <h4 className="text-base font-extrabold text-[#2D3748]">
                    {riskResult.headline}
                  </h4>
                  <p className="text-sm text-[#4A5568] leading-relaxed">
                    {riskResult.explanation}
                  </p>
                </div>

                {/* Sensed Probability Meters */}
                <div className="space-y-3 bg-white/70 rounded-2xl p-4 border border-black/5">
                  <div className="flex items-center justify-between text-xs font-bold text-[#4A5568]">
                    <span className="flex items-center gap-1.5">
                      <Camera className="w-3.5 h-3.5 text-[#5B8266]" /> Visual Threat P(human + vehicle)
                    </span>
                    <span className="font-extrabold">{(riskResult.camProb * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-amber-100 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${riskResult.camProb * 100}%` }}
                      transition={{ duration: 0.5, ease: 'easeOut' }}
                      className="h-full rounded-full bg-[#5B8266]"
                    />
                  </div>

                  <div className="flex items-center justify-between text-xs font-bold text-[#4A5568] pt-1">
                    <span className="flex items-center gap-1.5">
                      <Volume2 className="w-3.5 h-3.5 text-[#D96B43]" /> Acoustic Threat P(gunshot)
                    </span>
                    <span className="font-extrabold">{(riskResult.audProb * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-amber-100 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${riskResult.audProb * 100}%` }}
                      transition={{ duration: 0.5, ease: 'easeOut' }}
                      className="h-full rounded-full bg-[#D96B43]"
                    />
                  </div>
                </div>

                {/* Ranger Recommended Action */}
                <div className="p-4 rounded-2xl bg-white border border-black/5 space-y-1.5 shadow-xs">
                  <p className="text-xs font-extrabold uppercase tracking-wider text-[#5B8266]">
                    Recommended Ranger Protocol:
                  </p>
                  <p className="text-xs text-[#2D3748] font-medium leading-relaxed">
                    {riskResult.action}
                  </p>
                </div>

                {/* Rule Reference Footnote */}
                <div className="pt-2 border-t border-black/5 text-[11px] text-[#718096] flex items-center justify-between">
                  <span>Logic: 3-Tier Conservation Rule Head</span>
                  <span>Latency: ~18ms</span>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>

          {/* Quick Rules Summary Helper Card */}
          <div className="p-5 rounded-3xl bg-white border border-amber-100/90 shadow-sm space-y-2.5 text-xs text-[#4A5568]">
            <div className="flex items-center gap-1.5 font-extrabold text-[#2D3748]">
              <HelpCircle className="w-4 h-4 text-[#5B8266]" />
              <span>How Decision Tiers Are Formed:</span>
            </div>
            <ul className="space-y-1.5 leading-relaxed">
              <li className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-[#E05344]" />
                <span>
                  <strong>HIGH:</strong> Both signals agree (Human/Vehicle + Gunshot).
                </span>
              </li>
              <li className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-[#DD6B20]" />
                <span>
                  <strong>MEDIUM:</strong> Animal + Gunshot, Human/Vehicle + Background, or Empty + Gunshot.
                </span>
              </li>
              <li className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-[#38A169]" />
                <span>
                  <strong>LOW:</strong> Natural baseline (Animal or Empty + Background).
                </span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
