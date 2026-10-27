import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle2, ShieldAlert } from 'lucide-react';

export const RiskBadge = ({ level = 'LOW', size = 'md', showIcon = true, pulse = true }) => {
  const normLevel = (level || 'LOW').toUpperCase();

  const configs = {
    LOW: {
      bg: 'bg-[#EAF7EE]',
      border: 'border-[#A3E6B7]',
      text: 'text-[#22543D]',
      dot: 'bg-[#38A169]',
      glow: 'shadow-[0_0_12px_rgba(72,187,120,0.25)]',
      label: 'LOW RISK',
      sublabel: 'Normal forest activity',
      icon: CheckCircle2,
    },
    MEDIUM: {
      bg: 'bg-[#FFF5EA]',
      border: 'border-[#FBD38D]',
      text: 'text-[#7B341E]',
      dot: 'bg-[#DD6B20]',
      glow: 'shadow-[0_0_12px_rgba(221,107,32,0.25)]',
      label: 'MEDIUM RISK',
      sublabel: 'Investigate / verify',
      icon: AlertTriangle,
    },
    HIGH: {
      bg: 'bg-[#FFF0EE]',
      border: 'border-[#FEB2B2]',
      text: 'text-[#742A2A]',
      dot: 'bg-[#E05344]',
      glow: 'shadow-[0_0_14px_rgba(224,83,68,0.3)]',
      label: 'HIGH RISK',
      sublabel: 'Immediate patrol dispatch',
      icon: ShieldAlert,
    },
  };

  const config = configs[normLevel] || configs.LOW;
  const IconComponent = config.icon;

  const sizeClasses = {
    sm: 'px-2.5 py-1 text-xs gap-1.5',
    md: 'px-3.5 py-1.5 text-sm gap-2',
    lg: 'px-5 py-2.5 text-base gap-2.5 font-bold',
  };

  return (
    <motion.span
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className={`inline-flex items-center rounded-full font-semibold border ${config.bg} ${config.border} ${config.text} ${config.glow} ${sizeClasses[size]}`}
    >
      <span className="relative flex h-2.5 w-2.5">
        {pulse && normLevel !== 'LOW' && (
          <span
            className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${config.dot}`}
          />
        )}
        <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${config.dot}`} />
      </span>
      {showIcon && <IconComponent className={size === 'sm' ? 'w-3.5 h-3.5' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} />}
      <span>{config.label}</span>
    </motion.span>
  );
};
