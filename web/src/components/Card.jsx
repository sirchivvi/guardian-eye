import React from 'react';
import { motion } from 'framer-motion';

export const Card = ({
  children,
  className = '',
  hoverEffect = true,
  onClick = null,
  as = 'div',
  ...props
}) => {
  const Component = hoverEffect ? motion.div : 'div';
  const motionProps = hoverEffect
    ? {
        whileHover: { y: -4, transition: { duration: 0.2, ease: 'easeOut' } },
        whileTap: onClick ? { scale: 0.98 } : undefined,
      }
    : {};

  return (
    <Component
      onClick={onClick}
      className={`bg-white rounded-3xl p-6 sm:p-8 border border-amber-100/80 shadow-[0_8px_30px_rgb(0,0,0,0.03)] hover:shadow-[0_14px_35px_rgba(217,107,67,0.08)] transition-shadow ${onClick ? 'cursor-pointer' : ''} ${className}`}
      {...motionProps}
      {...props}
    >
      {children}
    </Component>
  );
};
