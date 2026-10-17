import { createRoot } from 'react-dom/client';
import { useEffect, useState } from 'react';
import App from './App';
import LandingPage from './LandingPage';
import './styles.css';

function Root() {
  const [page, setPage] = useState<'landing' | 'explorer'>(() =>
    window.location.hash === '#/explorer' ? 'explorer' : 'landing',
  );

  useEffect(() => {
    const onHash = () => {
      setPage(window.location.hash === '#/explorer' ? 'explorer' : 'landing');
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const goExplorer = () => {
    window.location.hash = '#/explorer';
    window.scrollTo(0, 0);
  };

  if (page === 'explorer') return <App />;
  return <LandingPage onEnterExplorer={goExplorer} />;
}

createRoot(document.getElementById('root')!).render(<Root />);
