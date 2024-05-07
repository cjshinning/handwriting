// // check
// setTimeout(() => {
//   console.log('setTimeout');
// })
// setImmediate(() => {
//   console.log('setImmediate');
// })

setImmediate(() => {
  console.log('setImmediate1');
  Promise.resolve('Promise microtask 1')
    .then(console.log);
});
setImmediate(() => {
  console.log('setImmediate2');
  Promise.resolve('Promise microtask 2')
    .then(console.log);
});