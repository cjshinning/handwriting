// commonjs模块用法
// var mod = require('./lib');
// console.log(mod.counter);   //3
// mod.incCounter();
// console.log(mod.counter);   //3

// es module模块用法
// import {counter, incCounter} from './lib';
// console.log(counter);   //3
// incCounter();
// console.log(counter);   //4

// exports.done = false;
// var lib = require('./lib');
// console.log('在main.js之中，lib.done = %j', lib.done);
// exports.done = true;
// console.log('main.js执行完毕');

// const counter = require('./lib');
// console.log(counter.count);

// const a1 = require('./lib');
// a1.foo = 2;

// const a2 = require('./lib');
// console.log(a2.foo);    //2
// console.log(a1 === a2); //true

// 多次require，只引入一次，然后缓存起来

// console.log(require.cache);

const a = require('./a');
console.log('in main, a.a1 = %j, a.a2 = %j', a.a1, a.a2);
// in b, a.a1 = true, a.a2 = undefined
// in a, b.done = undefined
// in main, a.a1 = true, a.a2 = true
