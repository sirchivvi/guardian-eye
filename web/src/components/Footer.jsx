import React from 'react';
import { Link } from 'react-router-dom';
import { Heart, Shield, Trees, ExternalLink } from 'lucide-react';
import { FoxSvg, ElephantSvg, OwlSvg, DeerSvg, SproutSvg } from './Animals';

export const Footer = () => {
  return (
    <footer className="mt-auto bg-gradient-to-b from-[#FFF9F0] to-[#F5ECE0] border-t border-amber-200/60 pt-16 pb-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 pb-12 border-b border-amber-200/60">
          {/* Brand & mission */}
          <div className="md:col-span-2 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-[#5B8266] p-1 flex items-center justify-center shadow-sm">
                <FoxSvg className="w-7 h-7" />
              </div>
              <span className="text-2xl font-extrabold text-[#2D3748] font-display">
                Guardian<span className="text-[#5B8266]">Eye</span>
              </span>
            </div>
            <p className="text-sm text-[#4A5568] max-w-md leading-relaxed">
              An open academic initiative creating intelligent, friendly multimodal sensing
              assistants for wildlife rangers. Combining camera-trap vision and forest bioacoustics
              to protect biodiversity safely.
            </p>
            <div className="flex items-center gap-3 pt-1">
              <div className="p-2 rounded-2xl bg-white/80 border border-amber-200/50 shadow-sm" title="Elephant">
                <ElephantSvg className="w-6 h-6" />
              </div>
              <div className="p-2 rounded-2xl bg-white/80 border border-amber-200/50 shadow-sm" title="Owl">
                <OwlSvg className="w-6 h-6" />
              </div>
              <div className="p-2 rounded-2xl bg-white/80 border border-amber-200/50 shadow-sm" title="Deer">
                <DeerSvg className="w-6 h-6" />
              </div>
              <div className="p-2 rounded-2xl bg-white/80 border border-amber-200/50 shadow-sm" title="Fox">
                <FoxSvg className="w-6 h-6" />
              </div>
            </div>
          </div>

          {/* Quick links */}
          <div>
            <h4 className="text-xs uppercase font-extrabold tracking-wider text-[#718096] mb-4">
              Explore App
            </h4>
            <ul className="space-y-2.5 text-sm font-semibold">
              <li>
                <Link to="/" className="text-[#4A5568] hover:text-[#5B8266] transition-colors flex items-center gap-1.5">
                  <Trees className="w-4 h-4 text-[#76A284]" /> Home & Overview
                </Link>
              </li>
              <li>
                <Link to="/try" className="text-[#4A5568] hover:text-[#D96B43] transition-colors flex items-center gap-1.5">
                  <span className="text-base">🎯</span> Interactive Risk Checker
                </Link>
              </li>
              <li>
                <Link to="/results" className="text-[#4A5568] hover:text-[#5B8266] transition-colors flex items-center gap-1.5">
                  <span className="text-base">📊</span> Benchmarks & Curves
                </Link>
              </li>
              <li>
                <Link to="/about" className="text-[#4A5568] hover:text-[#D96B43] transition-colors flex items-center gap-1.5">
                  <span className="text-base">🌱</span> Ethics & Dataset Limits
                </Link>
              </li>
            </ul>
          </div>

          {/* Research & Transparency */}
          <div>
            <h4 className="text-xs uppercase font-extrabold tracking-wider text-[#718096] mb-4">
              Research & Ethics
            </h4>
            <div className="p-4 rounded-2xl bg-[#FFFDF9] border border-amber-200/70 text-xs text-[#4A5568] space-y-2 shadow-sm">
              <div className="flex items-center gap-1.5 font-bold text-[#D96B43]">
                <Shield className="w-3.5 h-3.5" />
                <span>Honest Science</span>
              </div>
              <p className="leading-relaxed">
                Academic proof-of-concept tested on synthetic pairings. Not an operational field
                alarm system.
              </p>
              <Link
                to="/about"
                className="inline-flex items-center gap-1 font-bold text-[#5B8266] hover:underline pt-1"
              >
                Read honesty policy <ExternalLink className="w-3 h-3" />
              </Link>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between text-xs text-[#718096] gap-4">
          <p className="flex items-center gap-1.5">
            <span>Made with warm conservation care</span>
            <Heart className="w-3.5 h-3.5 text-[#E07A5F] fill-[#E07A5F]" />
            <span>for rangers & wildlife</span>
          </p>
          <p>© {new Date().getFullYear()} GuardianEye Research Project. Open-source conservation.</p>
        </div>
      </div>
    </footer>
  );
};
