import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = { title: "Yiti AI · Campus guide", description: "Find campus policy guidance with cited sources." };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
