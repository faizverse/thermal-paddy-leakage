const tf = require('@tensorflow/tfjs');
const fs = require('fs'); const jpeg = require('jpeg-js'); const path = require('path');
// file:// loader for node (pure JS backend)
const ioHandler = {
  load: async () => {
    const mj = JSON.parse(fs.readFileSync('/tmp/tfjs_model/model.json'));
    const bufs = mj.weightsManifest[0].paths.map(p => fs.readFileSync(path.join('/tmp/tfjs_model', p)));
    const all = Buffer.concat(bufs);
    return { modelTopology: mj.modelTopology, weightSpecs: mj.weightsManifest[0].weights,
             weightData: all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength),
             format: mj.format, generatedBy: mj.generatedBy, convertedBy: mj.convertedBy,
             signature: mj.signature, userDefinedMetadata: mj.userDefinedMetadata };
  }
};
function preprocess(img) { // identical logic to index.html
  return tf.tidy(() => {
    let x = tf.image.resizeBilinear(img.toFloat(), [224, 224], false, true);
    x = x.slice([0, 0, 0], [224 - Math.floor(224 * 0.15), 224 - Math.floor(224 * 0.20), 3]);
    x = tf.image.resizeBilinear(x, [224, 224], false, true);
    return x.expandDims(0);
  });
}
(async () => {
  await tf.setBackend('cpu');
  const model = await tf.loadGraphModel(ioHandler);
  
  let md=0;
  const b = JSON.parse(fs.readFileSync('raw120.json')); let agree = 0; const diffs=[]; const dis=[]; const t0=Date.now(); md = 0;
  for (let i = 0; i < b.paths.length; i++) {
    const raw = jpeg.decode(fs.readFileSync(b.paths[i]), { useTArray: true, formatAsRGBA: false });
    const img = tf.tensor3d(raw.data, [raw.height, raw.width, 3], 'int32');
    const p = model.predict(preprocess(img)).arraySync()[0];
    const am = p.indexOf(Math.max(...p)), ar = b.ref[i].indexOf(Math.max(...b.ref[i]));
    agree += (am === ar); p.forEach((v, j) => md = Math.max(md, Math.abs(v - b.ref[i][j]))); diffs.push(Math.max(...p.map((v,j)=>Math.abs(v-b.ref[i][j])))); if(am!==ar) dis.push(b.paths[i]); if(i%50==0) console.log(i, Date.now()-t0);
  }
  diffs.sort((x,y)=>x-y); console.log(`ALL ${b.paths.length}: argmax agree ${agree}/${b.paths.length}, max prob diff ${md.toFixed(4)}, p99 ${diffs[Math.floor(0.99*diffs.length)].toFixed(4)}, median ${diffs[Math.floor(0.5*diffs.length)].toFixed(4)}`); console.log('disagree', JSON.stringify(dis)); fs.writeFileSync('parity120.json', JSON.stringify({agree, n:b.paths.length, max:md, diffs, dis}));
})();
