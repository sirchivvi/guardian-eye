import React, { useEffect } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { Home } from './pages/Home';
import { TryIt } from './pages/TryIt';
import { ModelResults } from './pages/ModelResults';
import { AboutHonesty } from './pages/AboutHonesty';
import { NotFound } from './pages/NotFound';

// Automatically scroll to top on route change
const ScrollToTop = () => {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [pathname]);
  return null;
};

// Page transition container wrapper
const PageWrapper = ({ children }) => (
  <motion.div
    initial={{ opacity: 0, y: 12 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -12 }}
    transition={{ duration: 0.35, ease: 'easeOut' }}
    className="flex-1 flex flex-col"
  >
    {children}
  </motion.div>
);

export function App() {
  const location = useLocation();

  return (
    <div className="min-h-screen flex flex-col bg-[#FFF9F0] text-[#2D3748] font-sans antialiased selection:bg-[#FDE68A] selection:text-[#78350F]">
      <ScrollToTop />
      <Navbar />
      <main className="flex-1 flex flex-col">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route
              path="/"
              element={
                <PageWrapper>
                  <Home />
                </PageWrapper>
              }
            />
            <Route
              path="/try"
              element={
                <PageWrapper>
                  <TryIt />
                </PageWrapper>
              }
            />
            <Route
              path="/results"
              element={
                <PageWrapper>
                  <ModelResults />
                </PageWrapper>
              }
            />
            <Route
              path="/about"
              element={
                <PageWrapper>
                  <AboutHonesty />
                </PageWrapper>
              }
            />
            <Route
              path="*"
              element={
                <PageWrapper>
                  <NotFound />
                </PageWrapper>
              }
            />
          </Routes>
        </AnimatePresence>
      </main>
      <Footer />
    </div>
  );
}

export default App;
