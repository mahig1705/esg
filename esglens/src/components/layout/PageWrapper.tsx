import { motion } from "framer-motion";
import { ReactNode } from "react";

export function PageWrapper({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <motion.main
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={"ml-[72px] pt-8 min-h-screen relative " + className}
    >
      {children}
    </motion.main>
  );
}
