import React, { useState } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, Shield, Sparkles } from 'lucide-react';
import { FoxSvg, SproutSvg } from './Animals';

export const Navbar = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { to: '/', label: 'Home' },
    { to: '/try', label: 'Try Risk Checker' },
    { to: '/results', label: 'Model Results' },
    { to: '/about', label: 'About & Honesty' },
  ];

  return (
    <header className="sticky top-0 z-50 backdrop-blur-md bg-[#FFF9F0]/90 border-b border-amber-100/70 transition-colors">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo & Brand */}
          <Link
            to="/"
            className="flex items-center gap-3 group focus:outline-none focus:ring-2 focus:ring-[#5B8266] rounded-2xl p-1"
          >
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-[#76A284] to-[#4A6E55] p-1.5 shadow-sm group-hover:scale-105 transition-transform flex items-center justify-center">
              <FoxSvg className="w-8 h-8" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-xl sm:text-2xl font-extrabold tracking-tight text-[#2D3748] font-display">
                  Guardian<span className="text-[#5B8266]">Eye</span>
                </span>
                <SproutSvg className="w-4 h-4 text-[#5B8266] hidden sm:inline" />
              </div>
              <p className="text-xs text-[#718096] font-medium tracking-wide">
                Wildlife Conservation AI
              </p>
            </div>
          </Link>

          {/* Desktop Nav Links */}
          <nav className="hidden md:flex items-center gap-1 lg:gap-2">
            {navLinks.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === '/'}
                className={({ isActive }) =>
                  `px-4 py-2 rounded-full text-sm font-bold transition-all relative ${
                    isActive
                      ? 'text-[#5B8266] bg-[#E2ECE5]/70'
                      : 'text-[#4A5568] hover:text-[#2D3748] hover:bg-amber-100/50'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <span>{link.label}</span>
                    {isActive && (
                      <motion.div
                        layoutId="navPill"
                        className="absolute inset-0 rounded-full border border-[#C4D9CC] -z-10"
                        transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                      />
                    )}
                  </>
                )}
              </NavLink>
            ))}
          </nav>

          {/* Right Action CTA */}
          <div className="hidden md:flex items-center gap-3">
            <Link
              to="/try"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-[#D96B43] to-[#C2542B] text-white text-sm font-bold shadow-[0_4px_14px_rgba(217,107,67,0.3)] hover:shadow-[0_6px_20px_rgba(217,107,67,0.4)] hover:-translate-y-0.5 active:translate-y-0 transition-all"
            >
              <Sparkles className="w-4 h-4 text-[#FDE68A]" />
              <span>Interactive Demo</span>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2.5 rounded-2xl text-[#4A5568] hover:text-[#2D3748] hover:bg-amber-100/60 focus:outline-none focus:ring-2 focus:ring-[#5B8266]"
              aria-label="Toggle Navigation Menu"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden border-t border-amber-100 bg-[#FFF9F0] px-4 pt-3 pb-6 space-y-2 shadow-lg overflow-hidden"
          >
            {navLinks.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === '/'}
                onClick={() => setMobileMenuOpen(false)}
                className={({ isActive }) =>
                  `block px-4 py-3 rounded-2xl text-base font-bold transition-all ${
                    isActive
                      ? 'text-[#5B8266] bg-[#E2ECE5]'
                      : 'text-[#4A5568] hover:bg-amber-50'
                  }`
                }
              >
                {link.label}
              </NavLink>
            ))}
            <div className="pt-2">
              <Link
                to="/try"
                onClick={() => setMobileMenuOpen(false)}
                className="w-full flex items-center justify-center gap-2 px-5 py-3 rounded-2xl bg-[#D96B43] text-white font-bold text-center shadow-md"
              >
                <Sparkles className="w-4 h-4 text-[#FDE68A]" />
                <span>Try Risk Checker</span>
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};
