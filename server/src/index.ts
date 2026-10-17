import { createApp } from './app';
import { config } from './config';
import { initDb } from './db';

(async () => {
  await initDb();
  createApp().listen(config.port, () => console.log(`[api] :${config.port}  results=${config.resultsDir}`));
})();
