import React, { useEffect, useState, useRef } from 'react';
import { useInView } from 'framer-motion';

export const AnimatedCounter = ({
  target = 100,
  duration = 1.8,
  prefix = '',
  suffix = '',
  decimals = 0,
}) => {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });

  useEffect(() => {
    if (!isInView) return;

    let startTime = null;
    let animationFrameId;

    const animate = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / (duration * 1000), 1);
      // Gentle easeOutQuad
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      const currentVal = easedProgress * target;

      setCount(currentVal);

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(animate);
      } else {
        setCount(target);
      }
    };

    animationFrameId = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(animationFrameId);
  }, [isInView, target, duration]);

  const formatted = decimals > 0
    ? count.toFixed(decimals)
    : Math.round(count).toLocaleString();

  return (
    <span ref={ref} className="font-extrabold tracking-tight">
      {prefix}{formatted}{suffix}
    </span>
  );
};
