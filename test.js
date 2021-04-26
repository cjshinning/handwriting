// // function lastWordLength(str){
// //     var arr = str.split(' ')
// //     var str = arr.pop()
// //     console.log(str.length)
// // }
// // const str = 'hello nowcoder';
// // lastWordLength(str);

// // const readline = require('readline');

// // const rl = readline.createInterface({
// //     input: process.stdin,
// //     output: process.stdout
// // })

// // rl.question('你如何看待Node.js中文网？', (answer) => {
// //     console.log(`感谢您的宝贵意见：${answer}`);
// //     rl.close();
// // })

// // rl.on('line', (input) => {
// //     console.log(`接受到：${input}`);
// // })

// // rl.on('pause', () => {
// //     console.log('Readline 暂停');
// // })

// // 字符串的最后一个长度
// // var readline = require('readline');
// // const rl = readline.createInterface({
// //         input: process.stdin,
// //         output: process.stdout
// // });
// // rl.on('line', function(line){
// //    var arr = line.split(' ')
// //    var str = arr.pop()
// //    console.log(str.length)
// // }); 

// // var readline = require('readline');
// // const rl = readline.createInterface({
// //         input: process.stdin,
// //         output: process.stdout
// // });
// // rl.on('line', function(line){
// //     const arr = line.split('');
// //     for(let i = 0; i < i.length; i++){
// //         console.log(arr[i]);
// //     }
// // }); 

// var readline = require('readline');
// const rl = readline.createInterface({
//     input: process.stdin,
//     output: process.stdout
// });
// var lines = [];
// var lineIndex = 0;
// var filters = [];
// rl.on('line', (line)=> {
//     if (lineIndex === 0) {
//         lines.push(line);
//         lineIndex += 1;
//     } else {
//         lines.push(line);
//         filters = [...lines[0]].filter((item)=> {
//             return item.toLowerCase() === lines[1].toLowerCase()
//         })
//         lineIndex = 0;
//         console.log(filters.length);
//     }
// })


// const readline = require('readline');

// const rl = readline.createInterface({
//   input: process.stdin,
//   output: process.stdout
// });

// rl.on('line', function(line){
//     console.log(line.split('').reverse().join(''));
// })

// const readline = require('readline');

// const rl = readline.createInterface({
//     input: process.stdin,
//     out: process.stdout
// })

// rl.on('line', function(line){
//     console.log(eval(line));
// })

const reg = /.*/g;
console.log(reg.test('/'));
console.log(reg.test('/list'));
console.log(reg.test('/artilce/123'));