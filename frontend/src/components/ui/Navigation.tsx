import Link from 'next/link';

export default function Navigation() {
  return (
    <nav className="border-b border-sapphire-800/30 bg-black/40 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 group">
            <span className="text-2xl font-bold bg-gradient-to-r from-sapphire-400 to-cyan-400 bg-clip-text text-transparent">
              Real Time Fake Check
            </span>
          </Link>

          <div className="flex items-center gap-6">
            <Link
              href="/"
              className="text-gray-300 hover:text-sapphire-400 transition-colors font-medium"
            >
              Home
            </Link>
            <Link
              href="/dashboard"
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-sapphire-600 to-sapphire-500 text-white font-medium hover:shadow-lg hover:shadow-sapphire-500/50 transition-all"
            >
              Dashboard
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
