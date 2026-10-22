import { createRoot } from 'react-dom/client';
import { useEffect, useState } from 'react';
import App from './App';
import LandingPage from './LandingPage';
import GlobeDashboard from './globe/GlobeDashboard';
import './styles.css';

type Page = 'landing' | 'explorer' | 'globe';

function currentPage(): Page {
  if (window.location.hash === '#/explorer') return 'explorer';
  if (window.location.hash === '#/globe') return 'globe';
  return 'landing';
}

function Root() {
  const [page, setPage] = useState<Page>(currentPage);

  useEffect(() => {
    const onHash = () => setPage(currentPage());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const goExplorer = () => {
    window.location.hash = '#/explorer';
    window.scrollTo(0, 0);
  };

  const goGlobe = () => {
    window.location.hash = '#/globe';
    window.scrollTo(0, 0);
  };

  if (page === 'explorer') return <App />;
  if (page === 'globe') return <GlobeDashboard />;
  return <LandingPage onEnterExplorer={goExplorer} onEnterGlobe={goGlobe} />;
}

createRoot(document.getElementById('root')!).render(<Root />);
