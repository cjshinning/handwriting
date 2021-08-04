exports.a1 = true;
const b = require('./b');
console.log('in a, b.done = %j', b.done);
exports.a2 = true;