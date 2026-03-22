import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "KUru - AI Curriculum Advisor",
  description: "PLO-to-Career Navigator for Kasetsart University",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} h-full`}>
      <body className="min-h-full bg-ku-mint/40 font-sans text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}
