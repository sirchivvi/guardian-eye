import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Compass, Home as HomeIcon } from 'lucide-react';
import { OwlSvg, SproutSvg, FoxSvg } from '../components/Animals';

export const NotFound = () => {
  useEffect(() => {
    document.title = 'GuardianEye — Trail Not Found (404)';
  }, []);

  return (
    <div className="max-w-md mx-auto px-4 py-20 text-center space-y-6">
      <div className="relative inline-block">
        <div className="w-32 h-32 rounded-full bg-gradient-to-br from-[#FAF2E8] to-[#E2ECE5] mx-auto p-4 flex items-center justify-center shadow-sm">
          <OwlSvg className="w-20 h-20" />
        </div>
        <div className="absolute -bottom-2 -right-2 bg-white p-2 rounded-2xl border border-amber-200 shadow-md">
          <FoxSvg className="w-6 h-6" />
        </div>
      </div>

      <div className="space-y-2">
        <span className="text-xs font-black uppercase tracking-widest text-[#D96B43] bg-[#F9E5DE] px-3 py-1 rounded-full">
          404 Trail Lost
        </span>
        <h1 className="text-3xl font-extrabold text-[#2D3748] font-display">
          Looks like we wandered off the map!
        </h1>
        <p className="text-sm text-[#4A5568] leading-relaxed">
          The forest path you followed doesn't seem to exist. Even our wisest owls and foxes couldn't
          spot it. Let's head back to the main ranger station.
        </p>
      </div>

      <div className="pt-2">
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-6 py-3.5 rounded-full bg-[#5B8266] text-white font-extrabold text-sm shadow-md hover:bg-[#4A6E55] transition-all hover:scale-105 active:scale-100"
        >
          <HomeIcon className="w-4 h-4" />
          <span>Return to Safe Base (Home)</span>
        </Link>
      </div>
    </div>
  );
};
