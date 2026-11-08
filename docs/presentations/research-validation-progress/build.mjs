import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';

const root=path.dirname(fileURLToPath(import.meta.url));
const out=path.join(root,'output');
await fs.mkdir(out,{recursive:true});
const result=await build({entryPoints:[path.join(root,'src/main.jsx')],bundle:true,write:false,minify:true,format:'iife',platform:'browser',target:['es2020'],outfile:'deck.js',define:{'process.env.NODE_ENV':'"production"'},legalComments:'inline'});
const script=result.outputFiles.find(f=>f.path.endsWith('.js')).text;
const style=result.outputFiles.find(f=>f.path.endsWith('.css')).text;
const html=`<!doctype html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><meta name="description" content="Eight-slide progress presentation on the independent scientific audit and local recovery of cognitive radio spectrum sensing research."><title>Research Validation · Progress Update</title><style>${style}</style></head><body><div id="root"></div><noscript>This animated presentation requires JavaScript. It contains eight slides on audit milestones, four-layer corrections, calibrated mock results, native CNN validation, stress-test failures, confidence limits and the next RadioML experiment.</noscript><script>${script.replace(/<\/script/gi,'<\\/script')}</script></body></html>`;
await fs.writeFile(path.join(out,'Research_Validation_Progress.html'),html,'utf8');
await fs.copyFile(path.join(root,'src/evidence.json'),path.join(out,'evidence.json'));
console.log(`Built offline presentation: ${Buffer.byteLength(html).toLocaleString()} bytes`);
