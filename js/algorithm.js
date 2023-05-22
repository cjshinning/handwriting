// 视频学习：https://xiaochen1024.com/series/6196129fc1553b002e57bef5



// 题型：三道题是简单+简单+中等难度的题型。第一二题可能会是循环、数组、字符串、栈这些，第三题会难一点，二分查找、动态规划、DFS、BFS这些。 
// 参考资料：可看一下leetcode网的典型练习题目，编号如下：
// 字符串：3，49，30
// 线性表：86，16，27，732
// 队列：641，406，899
// 栈：946，116，117，895
// 哈希表：61，729，25，554
// dfs：105，112，98，494，547，1254
// bfs：1091，1129，102，101，752

// 题型：三道题是简单 + 简单 + 中等难度的题型。第一二题系统题比较多，可能会是循环、数组、字符串、栈这些，第三题会难一点，二分查找、动态规划、DFS、BFS这些。
// 一.练习用leetcode（有app可以手机平时看），https://leetcode.cn/
// 1. 系统题
// 精选题目：2043
// 练习题目：1268
// 2. 贪心
// 精选题目： 452
// 练习题目：621、376
// 3. BFS
// 精选题目： 127
// 练习题目： 139、130、317
// 4. DFS
// 精选题目： 934
// 练习题目：1102、533、531
// 5. 单调栈
// 精选题目： 1208
// 练习题目：209、3；
// 6. 字符串
// wa精选题目：5
// 练习题目：93、43

// leetcode 
// 一、字符串
// 3. 无重复字符的最长子串
var lengthOfLongestSubstring = function (s) {
  let left = 0;//定义左指针
  let res = 0;//定义无重复字符长度
  let map = new Map();//定义map方法，用以后续判断是否有重复字母、获取元素下标索引、存储索引
  for (let i = 0; i < s.length; i++) {//for遍历循环，i为右侧滑动指针
    if (map.has(s[i]) && map.get(s[i]) >= left)//如果字符中有重复的，并且右侧指针的索引>左侧指针索引
    {
      left = map.get(s[i]) + 1;//那么左侧指针索引进一位
    }
    res = Math.max(res, i - left + 1);//数学方法判断“符合题意”的字符最长值，res最初为0，通过不断循环迭代，来两者比较最长部分
    map.set(s[i], i);//每次循环更新一下map中的键值对，重点是i索引
  }//在不断判断与左侧滑动+max最长值判断的多重约束下，最终得到理想值res
  return res;//返回结果
};

// 30. 串联所有单词的子串
var findSubstring = function (s, words) {
  let wordlen = words[0].length;
  let ans = [];
  let sub = []
  words.sort();
  let str1 = words.toString()
  for (let i = 0; i < s.length; i++) {
    for (let j = 0; j < words.length; j++) {
      sub.push(s.substr(i + wordlen * j, wordlen))

    }
    sub.sort()
    if (sub.toString() === str1) {
      ans.push(i)
    }
    sub = []
  }
  return ans
};

// 49. 字母异位词分组
var groupAnagrams = function (strs) {
  const map = new Map();
  for (let str of strs) {
    let array = Array.from(str);//字符转成数组
    array.sort();//排序
    let key = array.toString();
    let list = map.get(key) ? map.get(key) : new Array();//从map中取到相应的数组
    list.push(str);//加入数组
    map.set(key, list);//重新设置该字符的数组
  }
  return Array.from(map.values());//map中的value转成数组
};

// 二、线性表
// 16. 最接近的三数之和
var threeSumClosest = function (nums, target) {
  let N = nums.length
  let res = Number.MAX_SAFE_INTEGER
  nums.sort((a, b) => a - b)
  for (let i = 0; i < N; i++) {
    let left = i + 1
    let right = N - 1
    while (left < right) {
      let sum = nums[i] + nums[left] + nums[right]
      if (Math.abs(sum - target) < Math.abs(res - target)) {
        res = sum
      }
      if (sum < target) {
        left++
      } else if (sum > target) {
        right--
      } else {
        return sum
      }
    }
  }
  return res
};

// 27. 移除元素
var removeElement = function (nums, val) {
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] == val) {
      nums.splice(i, 1);
      i--;
    }
  }
  return nums.length;
};

// 732. 我的日程安排表 III
var MyCalendarThree = function () {
  this.planTimes = [];
};

/** 
* @param {number} start 
* @param {number} end
* @return {number}
*/
MyCalendarThree.prototype.book = function (start, end) {
  if (this.planTimes.length === 0) {
    this.planTimes.push([start, 1], [end, -1]);
    return 1;
  }
  // 使用新数组来生成，可以保证在O(n)的时间下插入新的start和end
  let newPlan = [];
  // 因start和end的插入逻辑类似，故放入一个数组中表示待插入的节点，插入后会被删掉，也可换其它字段来标识start和end是否已被插入
  const arr = [
    [start, 1],
    [end, -1],
  ];
  this.planTimes.forEach((item, index) => {
    let i = 0;
    while (i < arr.length) {
      const curr = arr[i];
      if (item[0] === curr[0]) {
        item[1] += curr[1];
      } else if (item[0] > curr[0]) {
        newPlan.push(curr);
      }
      if (item[0] >= curr[0]) {
        arr.splice(i, 1);
      } else {
        i++;
      }
    }
    newPlan.push(item);
  });
  if (arr.length) {
    newPlan.push(...arr);
  }
  this.planTimes = newPlan;
  let res = 0;
  let count = 0;
  newPlan.forEach((item) => {
    count += item[1];
    res = Math.max(count, res);
  });
  return res;
};

// 三、队列
// 406. 根据身高重建队列
var reconstructQueue = function (people) {
  if (!people || !people.length) return [];
  people.sort((a, b) => a[0] === b[0] ? a[1] - b[1] : b[0] - a[0]);

  const res = [];
  people.forEach(item => {
    res.splice(item[1], 0, item); // 插入到k对应的位置
  })
  return res;
};

// 641. 设计循环双端队列
var MyCircularDeque = function (k) {
  // 队列的容量
  this.capacity = k;
  // 使用数组存放队列元素，所有的初始值都是-1，取值的时候直接返回
  this.queue = new Array(k).fill(-1);
  // 队列的头指针，即队列头元素的位置
  this.head = 0;
  // 队列的尾指针，即尾部要插入元素的位置，也就是队列的尾元素的位置+1
  this.tail = 0;
};

// 将index-1，需要考虑index到达数组首尾时需要循环
MyCircularDeque.prototype.reduceIndex = function (index) {
  return (index + this.capacity - 1) % this.capacity;
};

// 将index+1，需要考虑index到达数组首尾时需要循环
MyCircularDeque.prototype.addIndex = function (index) {
  return (index + 1) % this.capacity;
};

/**
 * Adds an item at the front of Deque. Return true if the operation is successful.
 * @param {number} value
 * @return {boolean}
 */
MyCircularDeque.prototype.insertFront = function (value) {
  // 判断队列是否已满
  if (this.isFull()) {
    return false;
  }

  // 从头部插入元素时，要先将头指针向前移动一位
  this.head = this.reduceIndex(this.head);
  // 在新的头指针位置插入元素
  this.queue[this.head] = value;

  return true;
};

/**
 * Adds an item at the rear of Deque. Return true if the operation is successful.
 * @param {number} value
 * @return {boolean}
 */
MyCircularDeque.prototype.insertLast = function (value) {
  // 判断队列是否已满
  if (this.isFull()) {
    return false;
  }

  // 在尾指针的位置插入元素
  this.queue[this.tail] = value;
  // 将尾指针向后移动一位，指向下一次插入元素的位置
  this.tail = this.addIndex(this.tail);

  return true;
};

/**
 * Deletes an item from the front of Deque. Return true if the operation is successful.
 * @return {boolean}
 */
MyCircularDeque.prototype.deleteFront = function () {
  // 判断队列是否为空
  if (this.isEmpty()) {
    return false;
  }

  // 将头指针的值置为-1，表示元素被删除
  this.queue[this.head] = -1;
  // 删除元素后，要将头指针向后移动一位
  this.head = this.addIndex(this.head);

  return true;
};

/**
 * Deletes an item from the rear of Deque. Return true if the operation is successful.
 * @return {boolean}
 */
MyCircularDeque.prototype.deleteLast = function () {
  // 判断队列是否为空
  if (this.isEmpty()) {
    return false;
  }

  // 先将尾指针向前移动一位，指向队尾元素
  this.tail = this.reduceIndex(this.tail);
  // 将队尾元素设置为-1
  this.queue[this.tail] = -1;

  return true;
};

/**
 * Get the front item from the deque.
 * @return {number}
 */
MyCircularDeque.prototype.getFront = function () {
  // 直接返回头指针的元素即可，由于初始值是-1，因此如果队列为空，会返回-1
  return this.queue[this.head];
};

/**
 * Get the last item from the deque.
 * @return {number}
 */
MyCircularDeque.prototype.getRear = function () {
  // 直接返回尾指针-1的元素即可，由于初始值是-1，因此如果队列为空，会返回-1
  return this.queue[this.reduceIndex(this.tail)];
};

/**
 * Checks whether the circular deque is empty or not.
 * @return {boolean}
 */
MyCircularDeque.prototype.isEmpty = function () {
  // 如果头尾指针的位置相同，且对应位置的值为-1，表示队列中已无元素，则为空
  return this.head === this.tail && this.queue[this.head] < 0;
};

/**
 * Checks whether the circular deque is full or not.
 * @return {boolean}
 */
MyCircularDeque.prototype.isFull = function () {
  // 如果头尾指针的位置相同，且对应位置的值不为-1，此时无法再插入元素，则队列已满
  return this.head === this.tail && this.queue[this.head] >= 0;
};

// 四、栈
// 116. 填充每个节点的下一个右侧节点指针
const connect = (root) => {
  if (root == null) {
    return root;
  }

  const dfs = (root) => {
    if (root.left == null && root.right == null) {
      return;
    }
    root.left.next = root.right;
    if (root.next) {
      root.right.next = root.next.left;
    }
    dfs(root.left);
    dfs(root.right);
  };

  dfs(root);
  return root;
};

// 117. 填充每个节点的下一个右侧节点指针 II
var connect = function (root) {
  if (!root) return null;

  const queue = [root];
  let cur;
  let level;

  while (queue.length) {
    level = queue.length;
    let i = 0;
    const temp = [];

    while (i++ < level) {
      cur = queue.shift();

      temp.push(cur);

      cur.left && queue.push(cur.left);
      cur.right && queue.push(cur.right);
    }

    temp.forEach((v, k) => {
      v.next = temp[k + 1] || null;
    });
  }

  return root;
};

// 895. 最大频率栈
var FreqStack = function () {
  // 记录 FreqStack 中元素的最大频率
  this.maxFreq = 0;
  // 记录 FreqStack 中每个 val 对应的出现频率，后文就称为 VF 表
  this.valToFreq = new Map();
  // 记录频率 freq 对应的 val 列表，后文就称为 FV 表
  this.freqToVals = new Map();
};

/**
 * @param {number} val
 * @return {void}
 */
FreqStack.prototype.push = function (val) {
  // 修改 VF 表：val 对应的 freq 加一
  let freq = (this.valToFreq.get(val) || 0) + 1;
  this.valToFreq.set(val, freq);
  // 修改 FV 表：在 freq 对应的列表加上 val
  let vals = this.freqToVals.get(freq) || [];
  vals.push(val);
  this.freqToVals.set(freq, vals);
  // 更新 maxFreq
  this.maxFreq = Math.max(this.maxFreq, freq);
};

/**
 * @return {number}
 */
FreqStack.prototype.pop = function () {
  // 修改 FV 表：pop 出一个 maxFreq 对应的元素 v
  let vals = this.freqToVals.get(this.maxFreq);
  let v = vals.pop();
  // 修改 VF 表：v 对应的 freq 减一
  let freq = this.valToFreq.get(v) - 1;
  this.valToFreq.set(v, freq);
  // 更新 maxFreq
  if (!vals.length) {
    // 如果 maxFreq 对应的元素空了
    this.maxFreq--;
  }
  return v;
};


// 946. 验证栈序列
var validateStackSequences = function (pushed, popped) {
  let stack = [];
  let i = 0;
  let j = 0;

  while (i < pushed.length) {
    stack.push(pushed[i]);
    while (stack[stack.length - 1] === popped[j] && stack.length) {
      j++;
      stack.pop();
    }
    i++
  }
  return stack.length === 0;
};

// 五、哈希表
// 25. K 个一组翻转链表
var reverseKGroup = function (head, k) {
  if (!head) return head;
  if (k < 2) return head;
  let newHead = head;
  const travel = (node, preNode) => {
    let i = 0;
    let stack = [];
    while (node && i < k) {
      stack.push(node);
      node = node.next;
      i++;
    }
    let cur = stack.pop();
    if (i === k) {
      if (newHead === head) newHead = cur;
      if (preNode) preNode.next = cur;
      while (stack.length) {
        cur.next = stack.pop();
        cur = cur.next;
      }
      cur.next = node;
    }
    if (node) {
      return travel(node, cur);
    }
  }
  travel(head, null);
  return newHead;
};

// 61. 旋转链表
var rotateRight = function (head, k) {
  if (!head || !head.next || !k) return head;
  let len = 1, cur = head;
  while (cur.next) {
    cur = cur.next;
    len++;
  }
  let move = len - k % len;
  cur.next = head;
  while (move) {
    cur = cur.next;
    move--;
  }
  let ans = cur.next;
  cur.next = null;
  return ans;
};

// 554. 砖墙
var leastBricks = function (wall) {
  let map = {}
  let min = 0
  for (let i = 0; i <= wall.length - 1; i++) {
    let sum = 0
    for (let j = 0; j < wall[i].length - 1; j++) {
      sum += wall[i][j]
      !map[sum] && (map[sum] = 0);
      map[sum] += 1
      min = Math.max(map[sum], min)
    }
  }
  return wall.length - min
};

// 729. 我的日程安排表 I
var MyCalendar = function () {
  this.schedule = [];
};

/** 
 * @param {number} start 
 * @param {number} end
 * @return {boolean}
 */
MyCalendar.prototype.book = function (start, end) {
  const schedule = this.schedule;
  const len = schedule.length;

  if (len === 0) {
    schedule.push([start, end]);
    return true;
  }

  let l = 0, r = len;
  while (l < r) {
    let m = l + ((r - l) >> 1);
    if (schedule[m][1] <= start) {
      l = m + 1;
    } else if (schedule[m][1] > start) {
      r = m;
    }
  }

  let idx = l - 1; // schedule 中小于等于插入区间 start 右边界
  let insertIndex; // 最后插入区间的位置

  if (idx === -1) {
    if (schedule[0][0] >= end) {
      /* 如果 end 比最小区间左边界还要小，则插入到最左边 */
      insertIndex = 0;
    }
  } else {
    /* 如果 start 大于等于 「schedule 中小于等于插入区间 start 右边界」的 end
       并且 end 小于下一个区间的 start 或者后面没有区间了，则可以插入 */
    if (start >= schedule[idx][1] &&
      (idx + 1 === len || end <= schedule[idx + 1][0])
    ) {
      insertIndex = idx + 1;
    }
  }

  if (insertIndex !== undefined) {
    schedule.splice(insertIndex, 0, [start, end]);
    return true;
  }

  return false;
};

// 六、dfs
// 98. 验证二叉搜索树
// 限定以 root 为根的子树节点必须满足 max.val > root.val > min.val
var isValidBST = function (root, min = -Infinity, max = Infinity) {
  // 如果是空节点
  if (!root) return true;
  // 当前节点的值大于最小值，小于最大值；（换句话说，当前节点的值大于左子树所有节点的值，小于右子树中所有节点的值 ）
  // 限定左子树的最大值是 root.val，右子树的最小值是 root.val
  return (
    root.val > min &&
    root.val < max &&
    isValidBST(root.left, min, root.val) &&
    isValidBST(root.right, root.val, max)
  );
};

// 105. 从前序与中序遍历序列构造二叉树
const buildTree = (preorder, inorder) => {
  const map = new Map();
  for (let i = 0; i < inorder.length; i++) {
    map.set(inorder[i], i);
  }
  const helper = (p_start, p_end, i_start, i_end) => {
    if (p_start > p_end) return null;
    let rootVal = preorder[p_start];    // 根节点的值
    let root = new TreeNode(rootVal);   // 根节点
    let mid = map.get(rootVal);         // 根节点在inorder的位置
    let leftNum = mid - i_start;        // 左子树的节点数
    root.left = helper(p_start + 1, p_start + leftNum, i_start, mid - 1);
    root.right = helper(p_start + leftNum + 1, p_end, mid + 1, i_end);
    return root;
  };
  return helper(0, preorder.length - 1, 0, inorder.length - 1);
};

// 112. 路径总和
var hasPathSum = function (root, targetSum) {
  if (!root) return 0
  let res = false
  const dfs = (root, sum) => {
    if (!root) return
    if (sum == targetSum && (!root.left && !root.right)) {
      res = true
    }
    if (root.left) dfs(root.left, sum + root.left.val)
    if (root.right) dfs(root.right, sum + root.right.val)
  }
  dfs(root, root.val)
  return res
};

// 494. 目标和
var findTargetSumWays = function (nums, target) {
  const sum = nums.reduce((p, v) => p + v);
  if (Math.abs(target) > sum) return 0;
  if ((target + sum) % 2) return 0;
  const left = (target + sum) / 2;
  let dp = new Array(left + 1).fill(0);
  dp[0] = 1;
  for (let i = 0; i < nums.length; i++) {
    for (let j = left; j >= nums[i]; j--) {
      dp[j] += dp[j - nums[i]];
    }
  }
  return dp[left];
};

// 547. 省份数量
var findCircleNum = function (isConnected) {
  const rows = isConnected.length;
  const visited = new Set();//记录是否访问过
  let count = 0;//省份数量
  for (let i = 0; i < rows; i++) {
    if (!visited.has(i)) {//如果没访问过
      dfs(isConnected, visited, rows, i);//深度优先遍历
      count++;//省份数量+1
    }
  }
  return count;
};

const dfs = (isConnected, visited, rows, i) => {
  for (let j = 0; j < rows; j++) {
    if (isConnected[i][j] == 1 && !visited.has(j)) {//如果i，j相连接
      visited.add(j);
      dfs(isConnected, visited, rows, j);//递归遍历
    }
  }
};

// 1254. 统计封闭岛屿的数目
var closedIsland = function (grid) {
  let ans = 0
  const A = grid
  const [row, col] = [A.length, A[0].length]
  const D = [[-1, 0], [0, -1], [1, 0], [0, 1]]
  for (let i = 0; i < row; i++) {
    for (let j = 0; j < col; j++) {
      if (A[i][j] === 0) BFS(i, j)
    }
  }
  return ans

  function BFS(i, j) {
    const q = []
    q.push([i, j])
    A[i][j] = -1
    let closed = true
    while (q.length > 0) {
      const n = q.length
      for (let _ = 0; _ < n; _++) {
        const [x, y] = q.pop()
        if (onBoundary(x, y)) closed = false
        for (const [dx, dy] of D) {
          const [nx, ny] = [x + dx, y + dy]
          if (valid(nx, ny) && A[nx][ny] === 0) {
            q.push([nx, ny])
            A[nx][ny] = -1
          }
        }
      }
    }
    if (closed) ans++
  }

  function valid(x, y) {
    return x >= 0 && x < row && y >= 0 && y < col
  }
  function onBoundary(x, y) {
    return x == 0 || x == row - 1 || y == 0 || y == col - 1
  }
};

// 七、bfs
// 101. 对称二叉树
var isSymmetric = function (root) {
  if (root == null) {
    return true;
  }
  return dfs(root.left, root.right)
}
const dfs = function (Left, Right) {
  if (Left == null && Right == null) {
    return true;
  }
  if (Left == null || Right == null || Left.val !== Right.val) {
    return false;
  }
  return dfs(Left.left, Right.right) && dfs(Left.right, Right.left)
}

// 102. 二叉树的层序遍历
var levelOrder = function (root) {
  const ret = [];
  if (!root) {
    return ret;
  }

  const q = [];
  q.push(root);//初始队列
  while (q.length !== 0) {
    const currentLevelSize = q.length;//当前层节点的数量
    ret.push([]);//新的层推入数组
    for (let i = 1; i <= currentLevelSize; ++i) {//循环当前层的节点
      const node = q.shift();
      ret[ret.length - 1].push(node.val);//推入当前层的数组
      if (node.left) q.push(node.left);//检查左节点，存在左节点就继续加入队列
      if (node.right) q.push(node.right);//检查左右节点，存在右节点就继续加入队列
    }
  }

  return ret;
};

// 752. 打开转盘锁
var openLock = function (deadends, target) {
  const deads = new Set()
  for (const ds of deadends) {
    deads.add(ds)
  }
  const queue = new Queue()
  const visited = new Set()
  // 从起点开始广度优先搜索
  let step = 0
  queue.enqueue('0000')
  visited.add('0000')

  while (!queue.isEmpty()) {
    let size = queue.size()
    // 遍历当前节点后的所有可能转法
    for (let i = 0; i < size; i++) {
      const curr = queue.dequeue()

      // 跳过死亡数字
      if (deads.has(curr)) {
        continue
      }
      if (curr === target) {
        return step
      }

      // 开转
      for (let j = 0; j < 4; j++) {
        const plus = plusOne(curr, j)
        if (!visited.has(plus)) {
          queue.enqueue(plus)
          visited.add(plus)
        }

        const minus = minusOne(curr, j)
        if (!visited.has(minus)) {
          queue.enqueue(minus)
          visited.add(minus)
        }
      }
    }
    // 一层遍历完了
    step++
  }
  return -1
};

function plusOne(str, index) {
  let strArr = str.split('')
  const char = strArr[index]
  if (char === '9') {
    strArr[index] = '0'
  } else {
    strArr[index] = String(Number(char) + 1)
  }
  return strArr.join('')
}

function minusOne(str, index) {
  let strArr = str.split('')
  const char = strArr[index]
  if (char === '0') {
    strArr[index] = '9'
  } else {
    strArr[index] = String(Number(char) - 1)
  }
  return strArr.join('')
}

// 1091. 二进制矩阵中的最短路径
var shortestPathBinaryMatrix = function (grid) {
  const n = grid.length
  if (grid[0][0] !== 0 || grid[n - 1][n - 1] !== 0) return -1
  const direction = [[1, 1], [0, 1], [1, 0], [-1, 1], [1, -1], [0, -1], [-1, 0], [-1, -1]] // 八个方向
  const list = [[0, 0]]
  let count = 1
  while (list.length > 0) {
    const size = list.length // 当前层范围
    for (let i = 0; i < size; ++i) {
      const [curR, curC] = list.shift() // 取节点
      if (curR === n - 1 && curC === n - 1) return count // 找到最短路径
      for (let j = 0; j < 8; ++j) {
        const nextR = curR + direction[j][0], nextC = curC + direction[j][1]
        if (nextR < 0 || nextR >= n || nextC < 0 || nextC >= n ||
          grid[nextR][nextC] !== 0) continue // 索引不合法 或 值不为0 跳过
        grid[nextR][nextC] = 1
        list.push([nextR, nextC])
      }
    }
    ++count // 层数+1
  }
  return -1
};

// 1129. 颜色交替的最短路径
var shortestAlternatingPaths = function (n, red_edges, blue_edges) {
  const map = Array.from({ length: n }, () => []);
  for (const [f, t] of red_edges) {
    map[f].push(t);
  }
  for (const [f, t] of blue_edges) {
    map[f].push(t | 128);
  }
  const ans = new Array(n).fill(-1);
  const vis = new Set();
  let queue = [0, 128];
  let step = -1;
  while (queue.length) {
    const tmp = [];
    step++;
    for (const f of queue) {
      vis.add(f);
      const idx = f & -129;
      if (ans[idx] === -1) ans[idx] = step;
      for (const t of map[idx]) {
        if (vis.has(t)) continue;
        // 同色
        if ((f & 128) === (t & 128)) continue;
        tmp.push(t);
      }
    }
    queue = tmp;
  }
  return ans;
};



// 华为题库
// https://www.nowcoder.com/exam/oj/ta?tpId=37

// HJ1 字符串最后一个单词的长度
var readline = require("readline");
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  let array = line.split(" ");
  console.log(array[array.length - 1].length);
});

// HJ2 计算某字符出现次数
var readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
const lines = [];
rl.on("line", function (line) {
  lines.push(line);
  if (lines.length === 2) {
    const input = lines[0];
    const target = lines[1];
    var res = input.match(new RegExp(target, "gim"));
    if (res === null) console.log(0); else console.log(res.length)
  }
});

// HJ3 明明的随机数
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
const nums = []; // 存待排序的所有数字
const clearNums = []; // 存去重后的数字
rl.on("line", function (line) {
  nums.push(Number(line));
  if (nums.length - 1 === nums[0]) {
    nums.shift(); // 去掉首位表示输入数据数量的数字
    nums.forEach((f) => {
      if (!clearNums.includes(f)) {
        clearNums.push(f); //  已存在的数字不重复添加
      }
    });
    // 对去重之后的数据进行排序
    clearNums.sort((a, b) => {
      return a - b;
    });
    clearNums.forEach((f) => console.log(f));
  }
});

// HJ4 字符串分隔
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  let str = line + '00000000';
  for (let i = 8; i < str.length; i += 8) {
    console.log(str.substring(i - 8, i))
  }
})

// HJ5 进制转换
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  console.log(parseInt(line, 16));
});

// HJ6 质数因子
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  let num = Number(line);
  let res = [];
  for (let i = 2; i * i <= num; i++) {
    while (num % i === 0) {
      res.push(i);
      num /= i;
    }
  }
  if (num > 1) res.push(num);
  console.log(res.join(' ').trim())
})

// HJ7 取近似值
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  const num = parseFloat(line);
  console.log(Math.round(num))
})

// HJ8 合并表记录
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
let arr = []
rl.on('line', (line) => {
  arr.push(line)
})
rl.on('close', () => {
  const n = arr.shift();
  const nums = arr;
  let obj = {};
  nums.forEach(i => {
    let [k, v] = i.split(' ');
    if (obj[k]) {
      obj[k] += parseInt(v);
    } else {
      obj[k] = parseInt(v)
    }
  })
  for (let j in obj) {
    console.log(j + ' ' + obj[j])
  }
})


// HJ9 提取不重复的整数
const readline = require("readline");
//The readline module provides an interface(like a data structure) for reading data from a Readable stream (such as process.stdin) one line at a time.

const rl = readline.createInterface({
  // interface(like a data structure)
  input: process.stdin,
  //a readable strem
  output: process.stdout,
  //a wirtable strem
});

rl.on("line", function (line) {
  //’line’ is a listener event, when user press ENTER or RETURN will triger it
  var tokens = line.toString().split("").reverse();
  //input change into string and split to an array and reverse order of splited string
  var set = [...new Set(tokens)];
  // Set remove duplicate elements from the array
  console.log(Number(set.join("")));
  //make an array string
});

// HJ10 字符个数统计
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  let arr = line.split('');
  console.log(Array.from(new Set(arr)).length)
})

// HJ11 数字颠倒
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  const nums = line.split("").reverse().join("");
  console.log(nums);
});

// HJ12 字符串反转
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  console.log(line.split('').reverse().join(''));
});

// HJ13 句子逆序
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
let str = []
let list = [];
rl.on("line", function (line) {
  const tokens = line.split(" ");
  list.push(tokens);
  let [list1] = list;
  // console.log(list1);
  target = list1.reverse()
  console.log(...target)
});

// HJ14 字符串排序
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
let arr = [];
let num = 0;
rl.on('line', function (line) {
  arr.push(line);
  if (Number(arr[0]) === arr.length - 1) {
    arr.shift()
    arr.sort((a, b) => {
      return a > b ? 1 : -1
    });
    arr.forEach((data) => {
      console.info(data);
    })
  }
  //     
});


// HJ15 求int型正整数在内存中存储时1的个数
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})
rl.on('line', (line) => {
  console.log(parseInt(line, 10).toString(2).match(/1/g).length)
})

// HJ16 购物单
const readline = require("readline");
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

let condition = null;
let lines = [];
let result = [[]];

rl.on("line", (line) => {
  let values = line.split(" ").map(Number);
  if (!condition) {
    condition = ((monye, count) => {
      return { monye: monye, count: count };
    })(...values);
  } else {
    lines.push({
      price: values[0],
      importance: values[1],
      parent: values[2],
      children: [],
    });
  }
});

rl.on("close", () => {
  for (let i = 0; i < lines.length; i++) {
    let parent = lines[i].parent;
    if (parent > 0) {
      lines[parent - 1].children.push(i);
    }
  }
  for (let i = 0; i <= condition.monye; i++) {
    if (lines[0].price > i || lines[0].parent) {
      result[0][i] = 0;
      continue;
    }
    if (lines[0].children.length >= 1) {
      let children = lines[0].children;
      let parent_price = lines[0].price;
      let children_price = parent_price;
      let aswer_arr = [];
      aswer_arr.push(lines[0].price * lines[0].importance);
      for (let child of children) {
        if (lines[child].price + parent_price <= i) {
          aswer_arr.push(
            aswer_arr[0] +
            lines[child].price * lines[child].importance
          );
        }
        children_price += lines[child].price;
      }
      if (children_price <= i && lines[0].children.length > 1) {
        aswer_arr.push(
          lines[0].price * lines[0].importance +
          ((children, lines) => {
            let ret = 0;
            for (let child of children) {
              ret +=
                lines[child].price *
                lines[child].importance;
            }
            return ret;
          })(children, lines)
        );
      }
      result[0][i] = Math.max(...aswer_arr);
      continue;
    }
    result[0][i] = lines[0].price * lines[0].importance;
  }
  for (let i = 1; i < condition.count; i++) {
    for (let j = 0; j <= condition.monye; j++) {
      if (!result[i]) {
        result[i] = [];
      }
      if (lines[i].price > j || lines[i].parent) {
        result[i][j] = result[i - 1][j];
        continue;
      }
      if (lines[i].children.length >= 1) {
        let children = lines[i].children;
        let parent_price = lines[i].price;
        let children_price = parent_price;
        let aswer_arr = [];
        aswer_arr.push(
          result[i - 1][j - lines[i].price] +
          lines[i].price * lines[i].importance
        );
        for (let child of children) {
          let x = lines[child].price + parent_price;
          if (x <= j) {
            aswer_arr.push(
              result[i - 1][j - x] +
              lines[i].price * lines[i].importance +
              lines[child].price * lines[child].importance
            );
          }
          children_price += lines[child].price;
        }
        if (children_price <= j && lines[i].children.length > 1) {
          aswer_arr.push(
            result[i - 1][j - children_price] +
            lines[i].price * lines[i].importance +
            ((children, lines) => {
              let ret = 0;
              for (let child of children) {
                ret +=
                  lines[child].price *
                  lines[child].importance;
              }
              return ret;
            })(children, lines)
          );
        }
        result[i][j] = Math.max(result[i - 1][j], ...aswer_arr);
        continue;
      }
      result[i][j] = Math.max(
        result[i - 1][j],
        result[i - 1][j - lines[i].price] +
        lines[i].price * lines[i].importance
      );
    }
  }
  let ret = parseInt(result.slice(-1)[0].slice(-1)[0]);
  console.log(ret);
});


// HJ17 坐标移动

// HJ18 识别有效的IP地址和掩码并进行分类统计

// HJ19 简单错误记录
const readline = require("readline");
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

let arr = [];
rl.on("line", (line) => {
  arr.push(line);
});

rl.on("close", () => {
  effect(arr);
});

function effect(arr) {
  let map = new Map();
  for (let i = 0; i < arr.length; i++) {
    let str = arr[i].slice(
      arr[i].lastIndexOf("\\") + 1,
      arr[i].lastIndexOf(" ")
    );
    let num = arr[i].slice(arr[i].lastIndexOf(" ") + 1);
    if (str.length > 16) {
      str = str.slice(str.length - 16);
    }
    arr[i] = str + " " + num;
    // if(map.size>8){
    //     map.delete(arr[0])
    // }
    if (map.has(arr[i])) {
      map.set(arr[i], map.get(arr[i]) + 1);
    } else {
      map.set(arr[i], 1);
    }
  }
  for (const [key, value] of map) {
    if (map.size > 8) {
      map.delete(key);
    }
  }
  for (const [key, value] of map) {
    console.log(key, value);
  }
}

// HJ20 密码验证合格程序
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  console.log(isQualify(line) ? "OK" : "NG");
});

function isQualify(str) {
  let [correct, count] = [0, 0];
  if (str.length < 9 || str.includes("\n") || str.includes(" ")) return false;
  if (/[a-z]/.test(str)) {
    correct++;
  }
  if (/[A-Z]/.test(str)) {
    correct++;
  }
  if (/[0-9]/.test(str)) correct++;
  if (/[^\u4e00-\u9fa5a-zA-Z\d,\.，。]+/.test(str)) correct++;
  if (correct < 3) return false;

  const obj = {};
  for (let i = 0; i < str.length; i++) {
    let substring = str.substring(i, i + 3);
    if (substring.length < 3) continue;
    obj[substring] = null;
    count++;
  }
  return Object.keys(obj).length === count;
}


// HJ21 简单密码
const readline = require("readline");
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", (line) => {
  const arr = [
    "abc",
    2,
    "def",
    3,
    "ghi",
    4,
    "jkl",
    5,
    "mno",
    6,
    "pqrs",
    7,
    "tuv",
    8,
    "wxyz",
    9,
  ];
  let text = line;
  text = text.replace(/[a-z]/g, (a) => {
    for (let i = 0; i < arr.length; i++) {
      if (typeof arr[i] == "string" && arr[i].indexOf(a) != -1) {
        return arr[i + 1];
      }
    }
  });
  text = text.replace(/[A-Z]/g, (a) => {
    if (a == "Z") {
      return "a";
    } else {
      return String.fromCharCode(a.toLocaleLowerCase().charCodeAt(0) + 1);
    }
  });
  console.log(text);
});

// HJ22 汽水瓶
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  if (line !== '0') {
    console.log(Math.floor(parseInt(line) / 2))
  }
});

// HJ23 删除字符串中出现次数最少的字符
const readline = require("readline");
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", (line) => {
  let str = line;
  let result = line.split("").reduce((temp, data) => {
    temp[data] = temp[data] ? temp[data] + 1 : 1;
    return temp; //统计字母出现次数
  }, {});
  let min = 21;

  for (let index in result) {
    min = Math.min(min, result[index]);
  }
  for (let index in result) {
    if (min === result[index]) {
      let reg = new RegExp(index, "g");
      str = str.replace(reg, "");
    }
  }
  console.log(str);
});

// HJ24 合唱队


// HJ25 数据分类处理

// HJ26 字符串排序
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

rl.on("line", function (line) {
  let res = Array.from(line);
  let lineCopy = Array.from(line.replace(/[^a-z]/gim, "")).sort((a, b) => {
    let x = a.toLowerCase();
    let y = b.toLowerCase();
    return x < y ? -1 : x > y ? 1 : 0;
  });
  //     console.log('lineCopy :>> ', lineCopy);
  res.forEach((word, index) => {
    if (/[a-z]/gim.test(word)) {
      res[index] = lineCopy[0];
      lineCopy.shift();
    }
  });

  console.log(res.join(""));
});

// HJ27 查找兄弟单词
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
var inputs = [];
rl.on('line', function (line) {
  inputs = line.split(' ');
}).on('close', function () {
  let n = inputs[0];
  let candidate = inputs.slice(1, inputs.length - 2);
  let key = inputs[inputs.length - 2];
  let k = inputs[inputs.length - 1];
  let bro = searchBro(candidate, key);
  bro.sort(); //将兄弟单词按字典序排列
  console.log(bro.length);
  if (bro[k - 1]) {
    console.log(bro[k - 1]);
  }

})
//查找兄弟单词有哪些
function searchBro(arr, key) {
  let result = [];
  for (let i = 0; i < arr.length; i++) {
    //排序后比较字符串是否相等
    if ((arr[i] !== key) && ([...arr[i]].sort().join('') === [...key].sort().join('')))
      result.push(arr[i]);
  }
  return result;
}

// HJ28 素数伴侣
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
let arr = [];
rl.on("line", function (line) {
  arr.push(line);
});
rl.on("close", function () {
  let input = arr[1].split(" ");
  let odds = []; // 奇数
  let evens = []; //偶数
  let count = 0;

  let v = new Array(100); //标记访问
  let match = new Array(100); //标记与偶数链接的奇数

  //判断是否为质数
  function isPrime(num) {
    if (num < 2) {
      return false;
    }

    for (let i = 2; i * i <= num; i++) {
      if (num % i == 0) {
        return false;
      }
    }
    return true;
  }
  //判断是否匹配
  function isMatch(odd) {
    let evenLength = evens.length;
    for (let i = 0; i < evenLength; i++) {
      let sum = Number(odd) + Number(evens[i]);
      if (isPrime(sum) && !v[i]) {
        //有边，未访问，标记

        v[i] = true;

        //偶数未有匹配的奇数 或者右边有匹配
        if (!match[i] || isMatch(match[i])) {
          match[i] = odd;
          return true;
        }
      }
    }
    return false;
  }
  //执行
  function init() {
    //1.分奇数偶数
    //标记偶数是否匹配的对象
    input.forEach((c) => {
      if (c % 2 == 0) {
        evens.push(c);
      } else {
        odds.push(c);
      }
    });
    // 2.遍历奇数
    let oddLength = odds.length; //zuo
    for (let i = 0; i < oddLength; i++) {
      v = new Array(evens.length);
      let falg = isMatch(odds[i]);
      if (falg) {
        count++;
      }
    }
    console.log(count);
  }
  init();
});

// HJ29 字符串加解密
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

void (async function () {
  // 有限数据量，直接枚举出全部，查表就可以了
  let str1 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let str2 = "bcdefghijklmnopqrstuvwxyzaBCDEFGHIJKLMNOPQRSTUVWXYZA1234567890";
  let data = [];
  while ((line = await readline())) {
    data.push(line);
  }
  let arr1 = []; //解密后的字符集合；
  let arr2 = []; // 加密后的字符集合
  for (char of data[0]) {
    arr1.push(str2[str1.indexOf(char)]);
  }
  console.log(arr1.join(""));
  for (char of data[1]) {
    arr2.push(str1[str2.indexOf(char)]);
  }
  console.log(arr2.join(""));
})();

// HJ29 字符串加解密
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;
let lines = [];
void (async function () {
  // Write your code here
  while ((line = await readline())) {
    lines.push(line);
    if (lines.length == 2) {
      console.log(encryption(lines[0]));
      console.log(decryption(lines[1]));
    }
  }
})();

function encryption(str) {
  let arr = str.split("");
  for (let i = 0; i < arr.length; i++) {
    if (arr[i].match(/[A-Z]/)) {
      arr[i] =
        arr[i] == "Z"
          ? "a"
          : String.fromCharCode(
            arr[i].toLowerCase().charCodeAt(0) + 1
          );
    } else if (arr[i].match(/[a-z]/)) {
      arr[i] =
        arr[i] == "z"
          ? "A"
          : String.fromCharCode(
            arr[i].toUpperCase().charCodeAt(0) + 1
          );
    } else if (arr[i].match(/[0-9]/)) {
      arr[i] = arr[i] == "9" ? "0" : (parseInt(arr[i]) + 1).toString();
    }
  }
  let res = arr.join("");
  return res;
}

function decryption(str) {
  let arr = str.split("");
  for (let i = 0; i < arr.length; i++) {
    if (arr[i].match(/[A-Z]/)) {
      arr[i] =
        arr[i] == "A"
          ? "z"
          : String.fromCharCode(
            arr[i].toLowerCase().charCodeAt(0) - 1
          );
    } else if (arr[i].match(/[a-z]/)) {
      arr[i] =
        arr[i] == "a"
          ? "Z"
          : String.fromCharCode(
            arr[i].toUpperCase().charCodeAt(0) - 1
          );
    } else if (arr[i].match(/[0-9]/)) {
      arr[i] = arr[i] == "0" ? "9" : (parseInt(arr[i]) - 1).toString();
    }
  }
  let res = arr.join("");
  return res;
}

// HJ30 字符串合并处理
const readline = require("readline");
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  let arr = line.replace(' ', '').split('');
  let m = arr.length;
  let arr1 = []; arr2 = [];//用arr1和arr2存储偶数和奇数字符
  while (arr.length > 0) {
    arr1.push(arr.shift());
    arr2.push(arr.shift());
  }
  arr1.sort();
  arr2 = arr2.filter(Boolean).sort();//排序
  while (arr.length < m) {//用排序后的arr1和arr2重新组成字符串
    arr.push(arr1.shift());
    arr.push(arr2.shift());
  }
  arr = arr.filter(Boolean);//注意需要过滤掉空值
  let res = arr.map(x => //利用map()方法完成规定的字符转换
    x = parseInt(x, 16)
      ? parseInt(parseInt(x, 16).toString(2).padStart(4, '0').split('').reverse().join(''), 2).toString(16).toUpperCase()
      : x
  ).join('');
  console.log(res);
});

// HJ31 单词倒排
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  const word = line.replace(/[^\w+]/gi, ' ');
  const tokens = word.split(" ");
  var result = tokens.reverse().join(" ");
  console.log(result);
});

// HJ32 密码截取
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  //
  let lines = "$" + "#" + line.split("").join("#") + "#" + "&"; //一定奇数，中心为遍历到的数，index （两个index,同时向左右两边扩展），开头结尾数据不一样，一定不匹配
  let max = 0;
  for (let index in lines) {
    max = Math.max(max, center(lines, index, index))
  }
  console.log(max)
});
function center(arr, left, right) {
  let len = 0;
  while (arr[left] == arr[right]) {
    len = right - left + 1;
    left--;
    right++;
  }

  return parseInt(len / 2)
}


// HJ33 整数与IP地址间的转换
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

void (async function () {
  // Write your code here
  let lineArr = [];
  while ((line = await readline())) {
    lineArr.push(line);
  }
  let first = lineArr[0].split(".");
  let second = lineArr[1];
  // IP地址转十进制
  let ipStr = "";
  for (let i = 0; i < first.length; i++) {
    // toString(2)转二进制，padStart()方法用于补足字符串前面，直到指定长度
    let temp = Number(first[i]).toString(2).padStart(8, "0");
    ipStr += temp;
  }
  console.log(parseInt(ipStr, 2)); // 转为十进制
  // 十进制转IP地址
  let changeStr = Number(second).toString(2).padStart(32, "0");
  let changeArr = []; // 用于存储八位二进制的数组
  for (let i = 0; i < 32; i += 8) {
    changeArr.push(changeStr.slice(i, i + 8));
  }
  let strArr = changeArr.map((item) => parseInt(item, 2)); // 将数组的每一项转十进制
  console.log(strArr.join("."));
})();

// HJ34 图片整理
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  const tokens = line.split("");
  var res = tokens.sort().join('');
  console.log(res);
});


// HJ35 蛇形矩阵
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', function (line) {
  const n = parseInt(line);
  let res = new Array(n).fill().map(item => Array());
  let cur = 1;
  for (let i = 0; i < n; i++) {
    for (let j = i; j >= 0; j--) {
      res[j].push(cur);
      cur++;
    }
  }
  for (let line of res) {
    console.log(line.join(' '));
  }
});

// HJ36 字符串加密
const readline = require('readline')

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

var lineCount = 0;
var arr = [];
rl.on('line', (input) => {

  lines = input.trim();
  arr.push(lines)

  lineCount++;
  if (lineCount === 2) {
    let str = arr[1];
    let set = new Set();
    let res = []
    let strkey = arr[0]
    line = strkey.toUpperCase() + "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    line.split("").forEach((ele) => {
      set.add(ele);
    })
    let key = new Array(...set);
    for (var j = 0; j < arr[0].length; j++) {
      for (var i = 0; i < key.length; i++) {
        if (arr[0][j].toUpperCase() === key[i]) {
          key[i] = arr[0][j]
        }
      }
    }
    str.split("").forEach((ele) => {
      if (ele.match(/[A-Z]/)) {
        res.push(key[ele.toLowerCase().charCodeAt(0) - 97]);
      } else {
        res.push(key[ele.charCodeAt(0) - 97].toLowerCase());
      }
    })
    console.log(res.join(''));
    lineCount = 0;
    arr = [];
  }
})


// HJ37 统计每个月兔子的总数
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  console.log(getnum(line));
});

function getnum(num) {
  if (num < 3) {
    return 1
  } else {
    return getnum(num - 1) + getnum(num - 2);
  }
}

// HJ38 求小球落地5次后所经历的路程和第5次反弹的高度
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

void async function () {
  // Write your code here
  while (line = await readline()) {
    let h = parseInt(line);
    let distants = h;
    let rebounceH = h;
    for (let i = 0; i < 5; i++) {
      distants += h * Math.pow(0.5, i);
      rebounceH *= 0.5;
    }
    console.log(distants - rebounceH * 2);
    console.log(rebounceH);
  }
}()

// HJ39 判断两个IP是否属于同一子网
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

function isInvalidIP(ip) {
  //ip 地址为 [0-255]
  return ip.split(".").some((v) => {
    let part = parseInt(v);
    return part < 0 || part >= 256;
  });
}
function isInvalidMask(mask) {
  // 先判断是否在[0-255]之间
  if (isInvalidIP(mask)) {
    return true;
  }
  // 再判断是否是连续的1开头
  let m2 = [];
  mask.split(".").forEach((v) => {
    //每个部分转为2进制
    let l = parseInt(v).toString(2);
    let zero = "";
    //不足8位的用0补充在最开始部分
    if (l < 8) {
      for (let i = 0; i < 8 - l; i++) {
        zero += "0";
      }
      l = zero + l;
    }
    m2.push(l);
  });
  //m2为最终的2进制串
  m2 = m2.join("");
  // 第一个0出现的位置
  let i = m2.indexOf("0");
  // 最后一个1出现的位置
  let j = m2.lastIndexOf("1");
  // 第一个0出现的位置在最后一个1出现的位置的右侧即非法
  return i < j;
}
function isSameSubweb(ip1, ip2, mask) {
  // 直接把2个ip的子网计算出来再逐一比较
  let sub1 = [],
    sub2 = [];
  let a = ip1.split(".").map((v) => parseInt(v));
  let b = ip2.split(".").map((v) => parseInt(v));
  let m = mask.split(".").map((v) => parseInt(v));
  for (let i = 0; i < 4; i++) {
    sub1.push(a[i] & m[i]);
    sub2.push(b[i] & m[i]);
  }
  for (let i = 0; i < 4; i++) {
    if (sub1[i] !== sub2[i]) {
      return false;
    }
    return true;
  }
}
void (async function () {
  // Write your code here
  while ((line = await readline())) {
    let mask = line;
    line = await readline();
    let ip1 = line;
    line = await readline();
    let ip2 = line;
    if (isInvalidMask(mask) || isInvalidIP(ip1) || isInvalidIP(ip2)) {
      //非法
      console.log(1);
    } else {
      //合法
      if (isSameSubweb(ip1, ip2, mask)) {
        console.log(0);
      } else {
        console.log(2);
      }
    }
  }
})();

// HJ40 统计字符
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  let num = 0;
  let word = 0;
  let space = 0;
  let other = 0;
  for (i of line) {
    if (/[a-zA-Z]/.test(i)) {
      word++;
    } else if (/\s/.test(i)) {
      space++;
    } else if (/[0-9]/.test(i)) {
      num++;
    } else {
      other++;
    }
  }
  console.log(word);
  console.log(space);
  console.log(num);
  console.log(other);
});

// HJ41 称砝码
const readline = require("readline");
const lr = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
let total = 0,
  mArr = [],
  xArr = [];
let map = new Map();
lr.on("line", (line) => {
  total++;
  if (total === 2) {
    mArr = line.split(" ").map(Number);
  } else if (total === 3) {
    xArr = line.split(" ").map(Number);
  }
});

lr.on("close", () => {
  let fama = [];
  mArr.forEach((item, index) => {
    for (let i = 0; i < xArr[index]; i++) {
      fama.push(item);
    }
  });
  let set = new Set();
  set.add(0);
  fama.forEach((item) => {
    let temp = Array.from(set);
    for (let i = 0; i < temp.length; i++) {
      set.add(temp[i] + item);
    }
  });
  console.log(set.size);
});

// HJ42 学英语

// HJ43 迷宫问题
var readline = require("readline");

var rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

var args = [];

rl.on("line", (line) => {
  args.push(line);
});

rl.on("close", () => {
  var board = args
    .slice(1)
    .map((str) => str.split(" "))
    .map((arr) => arr.map(Number));

  var result = a(board);

  for (var i = 0; i < result.length; i++) {
    console.log(`(${result[i][0]},${result[i][1]})`);
  }
});

var directions = [
  // right
  [0, 1],
  // left
  [0, -1],
  // down
  [1, 0],
  // up
  [-1, 0],
];

/**
 * 数据范围： 2 <= n,m <= 10
 * 输入的内容只包含 0 <= val <= 1
 *
 * @param {number[][]} board - 迷宫地图，二维数组
 * @returns {number[][]}
 */
var a = function (board) {
  var track = [[0, 0]];
  var visited = getVisited(board);
  var result = [];

  backtrack(board, track, visited, result);

  return result;
};

/**
 * 回溯 - 迷宫走法
 * @param {numer[][]} board - 地图
 * @param {number[][]} track - 路径选择集合
 * @param {object} visited - 坐标是否被走过, 防止走回头路
 * @param {number[][]} result - 路径选择集合的结果
 */
var backtrack = function (board, track, visited, result) {
  var rows = board.length;
  var cols = board[0].length;

  var cur = track[track.length - 1];
  var row = cur[0];
  var col = cur[1];

  if (row === rows - 1 && col === cols - 1) {
    for (var m = 0; m < track.length; m++) {
      result.push([track[m][0], track[m][1]]);
    }
    return;
  }

  for (var direction of directions) {
    var [dy, dx] = direction;
    var row1 = row + dy;
    var col1 = col + dx;

    if (!testStep(board, row1, col1) || visited[row1][col1]) {
      continue;
    }

    visited[row1][col1] = true;

    track.push([row1, col1]);
    backtrack(board, track, visited, result);
    track.pop();
  }
};

function testStep(board, row, col) {
  var rows = board.length;
  if (row < 0 || row >= rows) {
    return false;
  }

  var cols = board[0].length;
  if (col < 0 || col >= cols) {
    return false;
  }

  var val = board[row][col];
  if (val === 1) {
    return false;
  }

  return true;
}

function getVisited(board) {
  var rows = board.length;
  var cols = board[0].length;
  var visited = new Array(rows).fill(null);

  for (var row = 0; row < rows; row++) {
    visited[row] = new Array(cols).fill(false);
  }

  return visited;
}

// HJ44 Sudoku
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
let data = [];
rl.on("line", function (line) {
  data.push(line.trim().split(" "));
});
rl.on("close", function () {
  resolveSudo();
  data.forEach((row) => {
    console.log(row.join(" "));
  });
});

function resolveSudo() {
  for (let i = 0; i < 9; i++) {
    for (let j = 0; j < 9; j++) {
      if (data[i][j] !== "0") continue;
      // 检测到了0 填充
      for (let k = 1; k <= 9; k++) {
        // 两个判断，数独是否有效，剩下的 0 是否能被填充
        if (isValidSodu(i, j, `${k}`)) {
          // 有效则填
          data[i][j] = `${k}`;
          // 并且满足剩下的数独是合理的
          if (resolveSudo()) {
            return true;
          }
          // 如果剩下的不满足，返回充填
          data[i][j] = "0";
        }
      }
      // 1-9 都试过了，都不行，那就返回false
      return false;
    }
  }
  // 遍历完了没有0了, 说明位置是合适的了
  return true;
}

function isValidSodu(i, j, k) {
  // 每一列不能重复
  for (let m = 0; m < 9; m++) {
    if (data[m][j] == k) return false;
  }
  // 每一行不能重复
  for (let n = 0; n < 9; n++) {
    if (data[i][n] == k) return false;
  }

  // 每一个 3 * 3 的九宫格也不能为0
  let rowStart = Math.floor(i / 3) * 3;
  let colStart = Math.floor(j / 3) * 3;
  for (let m = 0; m < 3; m++) {
    for (let n = 0; n < 3; n++) {
      if (data[rowStart + m][colStart + n] == k) return false;
    }
  }
  return true;
}

// HJ45 名字的漂亮度
var readline = require('readline');

var rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

var args = [];
var codea = ('a').charCodeAt(0);

rl.on('line', (line) => {
  args.push(line);
});

rl.on('close', () => {
  var arr = args.slice(1)
    .map((s) => getBeautyDegree(s.toLowerCase()));

  for (var i = 0; i < arr.length; i++) {
    console.log(arr[i]);
  }
});

function getBeautyDegree(str) {
  var arr = new Array(26).fill(0);
  var len = str.length;

  for (var i = 0; i < len; i++) {
    var ch = str[i];
    var index = ch.charCodeAt(0) - codea;
    arr[index] += 1;
  }

  arr.sort((a, b) => (b - a));

  var num = 26;
  var ret = 0;

  for (var i = 0; i < 26; i++) {
    if (arr[i] === 0) {
      break;
    }
    ret += num * arr[i];
    num -= 1;
  }

  return ret;
}

// HJ46 截取字符串
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
let arr = []
rl.on('line', (line) => {
  arr.push(line)
})
rl.on('close', () => {
  console.log(arr[0].slice(0, arr[1]))
})

// HJ48 从单向链表中删除指定值的节点
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

void (async function () {
  // Write your code here
  while ((line = await readline())) {
    let arr = line.split(" ").slice(1);
    let d = arr.pop();
    let res = [arr.shift()];
    for (let i = 1; i < arr.length; i += 2) {
      res.splice(res.indexOf(arr[i]) + 1, 0, arr[i - 1]);
    }
    res.splice(res.indexOf(d), 1);
    console.log(res.join(" "));
  }
})();

// HJ50 四则运算

const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  handle(line);
});

function handle(line) {
  let str = line.replace(/\[|{/g, "(").replace(/\]|}/g, ")");

  // 方法一：投机取巧法
  // console.log(eval(str))

  // 方法二：半投机取巧
  let stack = [...str];
  let arr = [];

  while (stack.length) {
    const stackCode = stack.pop();
    if (stackCode !== "(") {
      arr.unshift(stackCode);
    } else {
      let newStr = "";
      let arrCode = "";
      while (arrCode !== ")") {
        newStr += arrCode;
        arrCode = arr.shift();
      }
      arr.unshift(eval(newStr));
    }
  }

  console.log(eval(arr.join("")));
}

// HJ51 输出单向链表中倒数第k个结点
const readline = require("readline");
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

class NodeLink {
  constructor(val = undefined, next = null) {
    this.val = val;
    this.next = next;
  }
}

function link(sum, param, n) {
  let head = new NodeLink(param[0]);
  let cur = head;
  for (let i = 1; i < param.length; i++) {
    cur.next = new NodeLink(param[i]);
    cur = cur.next;
  }
  let i = head,
    j = head;
  while (n) {
    j = j.next;
    n--;
  }
  while (j) {
    j = j.next;
    i = i.next;
  }
  console.log(i.val);
}

const arr = [];
rl.on("line", function (line) {
  arr.push(line);
  if (arr.length === 3) {
    link(arr[0], arr[1].split(" "), arr[2]);
    arr.length = 0;
  }
});

// HJ52 计算字符串的编辑距离


// HJ53 杨辉三角的变形  
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  const n = Number(line);
  if (n == 1 || n == 2) {
    console.log("-1");
  } else {
    if (n % 4 == 1 || n % 4 == 3) {
      console.log("2");
    } else if (n % 4 == 2) {
      console.log("4");
    } else if (n % 4 == 0) {
      console.log("3");
    }
  }
});

// HJ55 挑7
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

void (async function () {
  // Write your code here
  while ((line = await readline())) {
    let n = parseInt(line);
    let arr = Array.from(Array(n + 1).keys());
    arr.shift();
    let res = [];
    for (let i = 0; i < arr.length; i++) {
      if (arr[i].toString().match(/7/g) || arr[i] % 7 == 0) {
        res.push(arr[i]);
      }
    }
    console.log(res.length);
  }
})();

// HJ56 完全数计算
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  //console.log(line);
  let num = line;
  let output = 0;
  for (num; num > 1; num--) {
    let a = Math.floor(num / 2);
    let arr = [];
    let res = 0;
    for (let i = 1; i <= a; i++) {
      if (num % i == 0) {
        arr.push(i);
      }
    }
    for (i in arr) {
      res += arr[i];
    }
    // console.log(res)
    if (res == num) {
      output++;
    }
  }
  console.log(output);
});

// HJ57 高精度整数加法
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
let arr = [];
rl.on("line", function (line) {
  let re = /[^0-9]/;
  if (!re.test(line) && 1 <= line.length < 10000) {
    arr.push(line);
  }

  if (arr.length == 2) {
    let res = BigInt(arr[0]) + BigInt(arr[1])
    console.log(res.toString())
  }
});

// HJ58 输入n个整数，输出其中最小的k个
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
let arr = []
rl.on('line', (line) => {
  arr.push(line);
})
rl.on('close', () => {
  let arr1 = arr[0].split(' ');
  let arr2 = arr[1].split(' ').sort((a, b) => a - b);
  let res = arr2.slice(0, arr1[1]).join(' ');
  console.log(res)
})

// HJ59 找出字符串中第一个只出现一次的字符
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  let strCounts = [];
  if (line.length > 0) {
    let res = "-1";
    for (let str of line) {
      let len = line.match(new RegExp(str, 'g')).length;
      if (len == 1) {
        res = str;
        break;
      }
    }
    console.log(res);
  }
});

// HJ60 查找组成一个偶数最接近的两个素数
const isPrime = (n) => {
  if (n >= 2) {
    for (let i = 2; i < n / 2; i++) {
      if (n % i === 0) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}
const judge = (n) => {
  for (let i = n / 2; i < n; i++) {
    let j = n - i;
    if (isPrime(i) && isPrime(j)) {
      console.log(j + '\n' + i);
      break;
    }
  }
}
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  ouput: process.stdout
})
rl.on('line', (line) => {
  return judge(line)
})

// HJ61 放苹果
const readline = require("readline");
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// 递归写法：
/*
  递归退出条件：
  苹果数只有1个，放哪个盘子都一样，方案为1
  苹果数为0，放置已经完成，方案为1
  盘子数为1，则手上的所有苹果都会放在这一个盘子中，方案为1
*/
// 如果盘子数大于苹果树：则必定有n-m个盘子空着，不会影响总的方案书，所以f(m,n)等价于f(m,m)
/*
  如果盘子数小于等于苹果数：则考虑是不是放满所有盘子。如果至少空一个盘子，则有f(m,n-1)；如果不要空盘子，则每个盘子至少放了一个苹果，放了不会影响方案总数，因此有f(m-n,n)个方案
*/

/*function solution(m,n){
  if(m === 0 || m === 1 || n === 1){
    return 1
  }else if(n > m){
    return solution(m,m)
  }else{
    return solution(m,n-1) + solution(m-n,n)
  }
}

rl.on('line',function(line){
  const arr = line.split(' ').map(item => +item)
  const res = solution(arr[0],arr[1])
  console.log(res)
})*/

/*
  动态规划：处理边缘节点（苹果数只有1个，放哪个盘子都一样，方案为1
  苹果数为0，放置已经完成，方案为1
  盘子数为1，则手上的所有苹果都会放在这一个盘子中，方案为1）
*/
// dp[i][j] 表示对于i个苹果和j个盘子有多少种放置方案
/*
 * 两种情况：
 * 当 i < j 时，苹果数量比盘子数量少，盘子一定会空，因此当前状态转移方程为 dp[i][j] = dp[i][i]；
 * 当 i >= j 时，苹果数量多于盘子，则考虑是不是每个盘子都装苹果，如果不装满盘子，至少有一个盘子不装，方案有 dp[i][j-1]；如果装满盘子，每个盘子中至少有一个苹果，
 * 相当于都去掉一个苹果后再分配，方案有 dp[i-j][j]。因此状态转移方程为 dp[i][j] = dp[i-j][j] + dp[i][j-1]
 *
 * 综上：状态转移方程为 dp[i][j] = {
 *   dp[i][i]    if(i<j)
 *   dp[i-j][j] + dp[i][j-1]  if(i>=j)
 *  }
 * */

function solution(m, n) {
  // j为0代表1个盘子，i为0代表0个苹果，i为1代表1个苹果（因为题目规定苹果可以为0，盘子最少是1）
  // 这里的j长度为n，i苹果数是加上了1的基础，j就是从0到n-1，所以在进行i、j代换时，需要有一个1的差值，不是很方便后续表达式代换
  const dp = new Array(m + 1).fill(0).map((item) => []);
  for (let j = 0; j < n; j++) {
    dp[0][j] = 1;
    dp[1][j] = 1;
  }
  for (let i = 0; i < m + 1; i++) {
    dp[i][0] = 1;
  }
  for (let i = 2; i < m + 1; i++) {
    for (let j = 1; j < n; j++) {
      if (i < j + 1) {
        dp[i][j] = dp[i][i - 1];
      } else {
        dp[i][j] = dp[i - j - 1][j] + dp[i][j - 1];
      }
    }
  }
  //   console.log(dp)
  console.log(dp[m][n - 1]);
}

/*
function solution(m,n){
  // j为1代表1个盘子，i为0代表0个苹果，i为1代表1个苹果。
  // 这里定义的j的长度为n+1，是与i苹果的数目一样在基础上加了1的，方便后面的表达式书写
  const dp = new Array(m+1).fill(0).map(item => [])
  for(let j=1;j<n+1;j++){
    dp[0][j] = 1
    dp[1][j] = 1
  }
  for(let i=0;i<m+1;i++){
    dp[i][1] = 1
  }
  for(let i=2;i<m+1;i++){
    for(let j=2;j<n+1;j++){
      if(i < j){
        dp[i][j] = dp[i][i]
      }else{
        dp[i][j] = dp[i-j][j] + dp[i][j-1]
      }
    }
  }
  // console.log(dp)
  console.log(dp[m][n])
}
*/

rl.on("line", function (line) {
  const arr = line.split(" ").map((item) => +item);
  solution(arr[0], arr[1]);
});




// aHJ62 查找输入整数二进制中1的个数
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  const arr = parseInt(line).toString(2).split('');
  console.log(arr.filter(v => v === '1').length)
})



//HJ63 DNA序列
const readline = require('readline')
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

function findStr(s, n) {
  const res = []
  for (let i = 0; i <= s.length - n; i++) {
    let tempStr = s.substr(i, n)
    let newTempStr = tempStr.replace(/[^CG]/g, '')
    let scale = newTempStr.length / n
    if (res.length === 0) {
      res[0] = scale
      res[1] = tempStr
    } else {
      if (scale > res[0]) {
        res[0] = scale
        res[1] = tempStr
      }
    }
  }
  console.log(res[1])
}

const arr = []
rl.on('line', function (line) {
  arr.push(line)
})
rl.on('close', function () {
  findStr(arr[0], +arr[1])
})




// HJ64 MP3光标位置
const readline = require('readline')
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

function test(n, str) {
  const arr = new Array(n).fill(0)
  let start = 0, end = n % 4 - 1, cur = 0
  for (const item of str) {
    if (item === 'U') {
      if (cur === 0) {
        cur = arr.length - 1
        if (n < 4) {
          start = 0
          end = n - 1
        } else {
          start = cur - 3
          end = cur
        }
      } else {
        cur--
        if (Math.abs(cur - end) > 3) {
          start = cur
          end = cur + 3
        }
      }
    } else {
      if (cur === arr.length - 1) {
        cur = 0
        start = 0
        end = 3
      } else {
        cur++
        if (Math.abs(cur - start) > 3) {
          end = cur
          start = end - 3
        }
      }
    }
  }
  let res = []
  for (let i = start + 1; i <= end + 1; i++) {
    res.push(i)
  }
  console.log(res.join(' '))
  console.log(++cur)
}

const arr = []
rl.on('line', function (line) {
  arr.push(line)
})
rl.on('close', function () {
  test(+arr[0], arr[1])
})




// HJ65 查找两个字符串a,b中的最长公共子串
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
let inputArr = [];//存放输入的数据
rl.on('line', function (line) {
  inputArr.push(line);
}).on('close', function () {
  console.log(maxString(inputArr[0], inputArr[1]))//调用函数并输出
})

function maxString(a, b) {
  if (a.length < b.length) {
    [a, b] = [b, a];
  }
  let res = '';

  for (let l = 0; l < b.length; l++) {
    for (let r = l + 1; r <= b.length; r++) {
      if (r - l > res.length && a.indexOf(b.slice(l, r)) !== -1) {
        res = b.slice(l, r);
      }
    }
  }
  return res;
}



HJ66 配置文件恢复
//1.s
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;
//立即执行函数
//就把命令按长度排序 （题目里已排好），然后正则表达式匹配，以开头匹配
let inputs = [
  "reset",
  "reset board",
  "board add",
  "board delete",
  "reboot backplane",
  "backplane abort",
];
let outputs = [
  "reset what",
  "board fault",
  "where to add",
  "no board at all",
  "impossible",
  "install first",
  "unknown command",
];
void (async function () {
  // Write your code here
  while ((line = await readline())) {
    let input = line.split(" ");
    if (input.length == 1) {
      let str = "^" + input[0];
      let re = new RegExp(str);
      if (re.test(inputs[0])) {
        console.log(outputs[0]);
      } else {
        console.log(outputs[6]);
      }
    } else if (input.length == 2) {
      let str1 = "^" + input[0];
      let str2 = "^" + input[1];
      // ]标记当前是否匹配
      let flag = false;
      let count = 0;
      let res = "";
      for (let i = 1; i < 6; i++) {
        let commands = inputs[i].split(" ");
        let com1 = commands[0];
        let com2 = commands[1];
        let re1 = new RegExp(str1);
        let re2 = new RegExp(str2);
        if (re1.test(com1)) {
          if (re2.test(com2)) {
            count++;
            res = outputs[i];
            flag = true;
          }
        }
      }
      if (flag && count == 1) {
        console.log(res);
      } else {
        console.log(outputs[6]);
      }
    }
  }
})();




// HJ67 24点游戏算法const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

rl.on('line', function (line) {
  let nums = line.split(' ').map(e => Number(e));
  let res = judgePoint24(nums);
  console.log(res);
});

function judgePoint24(nums) {
  let len = nums.length;
  if (len === 1) {
    return Math.abs(nums[0] - 24) < 0.000000001;
  }
  let isValid = false;
  for (let i = 0; i < len; i++) {
    for (let j = i + 1; j < len; j++) {
      const n1 = nums[i];
      const n2 = nums[j];
      const newNums = [];
      for (let k = 0; k < len; k++) {
        if (k !== i && k !== j) {
          newNums.push(nums[k]);
        }
      }
      isValid = isValid || judgePoint24([...newNums, n1 + n2]);
      isValid = isValid || judgePoint24([...newNums, n1 - n2]);
      isValid = isValid || judgePoint24([...newNums, n2 - n1]);
      isValid = isValid || judgePoint24([...newNums, n1 * n2]);
      if (n1 !== 0) {
        isValid = isValid || judgePoint24([...newNums, n2 / n1]);
      }
      if (n2 !== 0) {
        isValid = isValid || judgePoint24([...newNums, n1 / n2]);
      }

      if (isValid) {
        return true;
      }
    }
  }
  return false;
}




// J68 成绩排序
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
let result = [];
rl.on("line", function (line) {
  const tokens = line.split(" ");
  result.push(tokens);
});

rl.on("close", () => {
  result.shift();
  let sort = result.shift()[0];
  let sortArr = Boolean(Number(sort)) ? result.sort((x, y) => x[1] - y[1]) : result.sort((x, y) => y[1] - x[1]);
  sortArr.forEach(item => {
    let [key, value] = item;
    console.log(key, value)
  })
});



// HJ69 矩阵乘法
"use strict";
const readline = require("readline");
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

let matrix_1_row = 0;
let matrix_1_col = 0;
let matrix_2_row = 0;
let matrix_2_col = 0;
const matrix_1 = [];
const matrix_2 = [];

let lineNumber = 1;
let xLineNumber = 0;
let yLineNumber = 0;
rl.on('line', (line) => {
  if (lineNumber === 1) {
    matrix_1_row = parseInt(line);

    xLineNumber = matrix_1_row;
    lineNumber = lineNumber + 1;
  } else if (lineNumber === 2) {
    matrix_1_col = parseInt(line);
    matrix_2_row = matrix_1_col

    yLineNumber = matrix_2_row;
    lineNumber = lineNumber + 1;
  } else if (lineNumber === 3) {
    matrix_2_col = parseInt(line);

    lineNumber = lineNumber + 1;
  } else if (xLineNumber > 0) {
    matrix_1.push(line.split(" "));

    xLineNumber = xLineNumber - 1;
  } else if (yLineNumber > 0) {
    matrix_2.push(line.split(" "));

    yLineNumber = yLineNumber - 1;
    if (yLineNumber === 0) {
      const matrix_result = Array(matrix_1_row);

      (function loop(i) {
        matrix_result[i] = Array(matrix_2_col);
        (function loop(j) {
          matrix_result[i][j] = 0;
          (function loop(k) {
            matrix_result[i][j] = matrix_result[i][j] +
              matrix_1[i][k] * matrix_2[k][j]

            if (k === matrix_1_col - 1) return;
            return loop(k + 1);
          })(0);
          if (j === matrix_2_col - 1) return;
          return loop(j + 1);
        })(0);
        if (i === matrix_1_row - 1) return;
        return loop(i + 1);
      })(0);

      matrix_result.forEach((arr) => {
        console.log(arr.join(" "));
      });

      // next
      lineNumber = 1;
      matrix_1.length = 0;
      matrix_2.length = 0;
    }
  }
});




// HJ70 矩阵乘法计算量估算
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
let source = []
rl.on('line', function (line) {
  source.push(line.trim())
});

rl.on('close', function () {
  let n = Number(source.shift())
  let dict = 'ABCDEFGHIJKLMNO'
  let rule = source.pop()
  let idx = 0
  //     console.log(source)
  // 将规则按栈维护
  rule = rule.split('')
  let stack = []
  let res = 0
  for (let i = 0; i < rule.length; i++) {
    if (rule[i] == ")") {
      let b = stack.pop()
      let a = stack.pop()
      // 弹出括号
      stack.pop()
      // 累加结果
      res += a[0] * a[1] * b[1]
      // 封装新的值
      stack.push([a[0], b[1]])
    } else if (/[A-Z]/.test(rule[i])) {
      let idx = dict.indexOf(rule[i])
      //             console.log(idx)
      stack.push(source[idx].split(' '))
    } else {
      stack.push(rule[i])
    }
  }
  console.log(res)
})





// HJ71 字符串通配符
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
let count = 1
let s1, s2
rl.on('line', function (line) {
  if (count == 1) {
    s1 = line.trim()
  } else {
    s2 = line.trim()
  }
  count++
});

rl.on('close', function () {

  // 递归
  //     const dfs = (rule, target) => {
  //         if (rule == '' && target == '') return true
  //         else if (rule == '' && target !== '') return false
  //         else if (rule !== '' && target == '') return (rule.replace('*', '') == '')
  //         else {
  //             let match = false 
  //             match = (rule[0] == '?' && /\w/.test(target[0])) || rule[0].toLowerCase() == target[0].toLowerCase()
  //             if (rule[0] == '*') {
  //                 // 匹配 0 个 或者 匹配多个
  //                 return dfs(rule.slice(1), target) || dfs(rule, target.slice(1))
  //             } else {
  //                 return match &&  (dfs(rule.slice(1), target.slice(1)))
  //             }

  //         } 
  //     }
  //     console.log(dfs(s1, s2))        // 正则
  //     s1 = s1.replace(/\?/g, '[a-z0-9]').replace(/\*/g, '.*')
  //     console.log(new RegExp('^' + s1 + '$', 'i').test(s2))

  //动态规划
  let len1 = s1.length
  let len2 = s2.length
  let dp = new Array(len1 + 1).fill(0).map(_ => new Array(len2 + 1).fill(false))
  dp[0][0] = true

  if (s1[0] == "*") {
    dp[1][0] = true
  }

  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      if (s1[i - 1] == '?' && /[a-z0-9]/i.test(s2[j - 1])) {
        dp[i][j] = dp[i - 1][j - 1]
      }
      else if (s1[i - 1] == "*") {
        dp[i][j] = dp[i - 1][j] || dp[i - 1][j - 1] || dp[i][j - 1]
      }
      else {
        dp[i][j] = dp[i - 1][j - 1] && s1[i - 1].toLowerCase() == s2[j - 1].toLowerCase()
      }
    }
  }
  console.log(dp[len1][len2])
})





// HJ72 百钱买百鸡问题
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  for (let i = 0; i <= 20; i++) {
    for (let j = 0; j <= (100 - 5 * i) / 3; j++) {
      let k = 100 - i - j;
      if (i * 5 + j * 3 + k / 3 == 100 && k % 3 == 0) {
        console.log(i + ' ' + j + ' ' + k);
      }
    }
  }
});




// HJ73 计算日期到天数转换
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;
//
void async function () {
  // Write your code here

  while (line = await readline()) {
    //平年
    let arr = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    //润年
    let arr2 = [0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

    let tokens = line.split(' ');
    let year = parseInt(tokens[0]);
    let month = parseInt(tokens[1]);
    let day = parseInt(tokens[2]);
    let res = 0;
    if (isLeapYear(year)) {
      res = arr2[month] + parseInt(day)
    } else {
      res = arr[month] + parseInt(day)

    }
    console.log(res)
  }
}()
function isLeapYear(num) {
  if (num % 4 == 0 && num % 100 != 0 || num % 400 == 0) {
    return 1
  }
  return 0
}



// HJ74 参数解析
const rl = require("readline").createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on('line', (line) => {
  fun74(line.split(' '));
});
function fun74(ipt) {
  let i = 0, res = [];
  while (i < ipt.length) {
    // 如果当前字符串包含双引号（也就是当前字符串以双引号开头）
    if (ipt[i].includes('"')) {
      // 如果当前字符串的最后一个字符也是双引号，那就直接把双引号里边的内容 push 进最终结果中，continue
      if (ipt[i].charAt(ipt[i].length - 1) === '"') {
        res.push(ipt[i].substring(1, ipt[i].length - 1));
        i++;
        continue;
      } else {
        // 如果当前字符串不是以双引号结尾
        // 那就定义一个 tempRes
        let tempRes = [];
        // push 进当前字符串除了双引号外的内容
        tempRes.push(ipt[i].substring(1));
        i++;
        // 一直 push，直到遇到包含双引号的字符串（即与之相匹配的那个双引号所在的字符串）
        while (i < ipt.length && !ipt[i].includes('"')) {
          tempRes.push(ipt[i]);
          i++;
        }
        // 把这个包含双引号的字符串的除了双引号的部分 push 进 tempRes 中
        tempRes.push(ipt[i].substring(0, ipt[i].length - 1));
        res.push(tempRes.join(' ')); // 添加进最终的 res
      }
    } else {
      // 如果不含双引号，那就直接 push 进 res 中
      res.push(ipt[i]);
    }
    i++;
  }
  // 最后要在 res 的首端添加参数的数量，然后依次输出即可
  res.unshift(res.length);
  for (const e of res) {
    console.log(e);
  }
}




// HJ75 公共子串计算
const findCommonChildStr = (s) => {
  let s1 = s[0],
    s2 = s[1];
  for (let i = s1.length; i >= 0; i--) {
    for (let j = 0; j + i <= s1.length; j++) {
      if (s2.indexOf(s1.substring(j, j + i)) > -1) {
        return s1.substring(j, j + i).length;
      }
    }
  }
  return 0;
}
let arr = []
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  arr.push(line)
})
rl.on('close', () => {
  console.log(findCommonChildStr(arr))
})




// HJ76 尼科彻斯定理
const count = (m) => {
  let first = m * m - m + 1;
  let result = first;
  for (let i = 1; i < m; i++) {
    first = first + 2;
    result += '+' + String(first);
  }
  return result;
}
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  console.log(count(line))
})



// HJ77 火车进站
// 多行输入
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
const inputArr = [];//存放输入的数据
rl.on('line', function (line) {
  //line是输入的每一行，为字符串格式
  inputArr.push(line.split(' ').map(item => Number(item)));//将输入流保存到inputArr中（用map将字符串数组转为数字）
}).on('close', function () {
  let res = fun(inputArr)//调用函数
  res.forEach(item => console.log(item.join(' '))) //打印结果
})

//解决函数
function fun(arr) {
  let N = arr[0][0];
  let train = arr[1].concat();    //未进站的火车
  let res = [];    //输出的结果
  let station = []    //站台栈内的火车
  let outStation = []    //发出的火车
  let backtrace = (train, station, outStation) => {
    if (outStation.length == N) {
      res.push([...outStation])
      return;
    }
    if (station.length == 0 && train.length != 0) {
      // 如果车站为空且未进站的火车不为空，则进站
      station.push(train.shift())
      backtrace(train, station, outStation)
    } else if (station.length != 0 && train.length == 0) {
      // 如果车站不为空，且没有未进站的火车，则顺序出站
      outStation.push(station.pop())
      backtrace(train, station, outStation)
    } else if (station.length != 0 && train.length != 0) {
      // 如果车站不为空，且还有未进站的火车，则可以选择1.出站，2.进站,并回溯
      let temp1 = [...outStation];
      let temp2 = [...station];
      let temp3 = [...train]
      // 出站
      outStation.push(station.pop())
      backtrace(train, station, outStation)
      // 回溯 
      outStation = temp1;
      station = temp2;
      train = temp3;
      // 进站
      station.push(train.shift())
      backtrace(train, station, outStation)
    }
  }
  backtrace(train, station, outStation)
  return res.sort((a, b) => a.join('') - b.join(''))
}



// HJ80 整型数组合并
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

void async function () {
  // Write your code here
  const lines = [];
  while (line = await readline()) {
    lines.push(line);
    if (lines.length == 2) {
      var x1 = lines[1].split(' ');
    }
    if (lines.length == 4) {
      var x2 = lines[3].split(' ');
    }
  }
  let sum = x1.concat(x2);
  sum.sort((a, b) => a - b);
  for (let i = 1; i < sum.length; i++) {
    if (sum[i] == sum[i - 1]) {
      sum[i - 1] = '';
    }
  }
  console.log(sum.join(''));
}()



// HJ81 字符串字符匹配
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

void async function () {
  // Write your code here
  while (line = await readline()) {
    let s1 = line; //保存短串
    line = await readline();
    let s2 = line; //保存长串
    let included = s1.split('').every((c) => {
      return s2.includes(c);
    })
    console.log(included);
  }
}()


// HJ82 将真分数分解为埃及分数



// HJ83 二维数组操作



// HJ84 统计大写字母个数
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  console.log(line.match(/[A-Z]/g).length || 0)
})



// HJ85 最长回文子串
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

let arr = []
//回马车


void async function () {
  // Write your code here
  while (line = await readline()) {
    let str = "%" + "#" + line.split("").join("#") + "#" + "$"
    console.log(center(str))
  }
}()
function center(str) {
  let len = str.length;
  let max = 0;
  for (let i = 2; i < len - 1; i++) {
    let index = i - 1;
    let endex = i + 1;
    let length = 0;
    while (str[index] == str[endex]) {
      length = endex - index + 1
      index--;
      endex++;
    }
    max = Math.max(max, length)
  }
  return parseInt(max / 2)
}




// HJ86 求最大连续bit数
const getBigestBit = (num) => {
  const bitnum = Number(num).toString(2);
  const arr = bitnum.split('')
  let l = 0, r = 0;
  let max = 0;
  let len = arr.length;
  while (l < len && r < len) {
    if (arr[l] === '1' && arr[r] === '1') {
      max = Math.max(max, r - l + 1);
    } else {
      l = r + 1;
    }
    r++;
  }
  return max;
}

const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  console.log(getBigestBit(line))
})



// HJ87 密码强度等级
const countScore = (password) => {
  let score = 0;// 总得分
  let upper = false;// 大写字符是否存在
  let lower = false;// 小写字符是否存在
  let num = 0;// 数字几个
  let char = 0;// 符号几个
  for (let i of password) {
    if (i >= 'A' && i <= 'Z') {
      upper = true;
    } else if (i >= 'a' && i <= 'z') {
      lower = true;
    } else if (i >= '0' && i <= '9') {
      num += 1;
    } else {
      char += 1;
    }
  }
  // 密码长度
  if (password.length <= 4) {
    score += 5;
  } else if (password.length <= 7) {
    score += 10;
  } else {
    score += 25;
  }
  // 字母
  if (upper && lower) {
    score += 20;
  } else if ((upper && !lower) || (!upper && lower)) {
    score += 10
  }
  // 数字
  if (num === 1) {
    score += 10;
  } else if (num > 1) {
    score += 20;
  }
  // 符号
  if (char === 1) {
    score += 10;
  } else if (char > 1) {
    score += 25;
  }
  // 奖励
  if (upper && lower && num && char) {
    score += 5
  } else if ((upper || lower) && num && char) {
    score += 3;
  } else if (num && char) {
    score += 2;
  }
  if (score >= 90) {
    return "VERY_SECURE";
  } else if (score >= 80) {
    return "SECURE";
  } else if (score >= 70) {
    return "VERY_STRONG";
  } else if (score >= 60) {
    return "STRONG";
  } else if (score >= 50) {
    return "AVERAGE";
  } else if (score >= 25) {
    return "WEAK";
  } else {
    return "VERY_WEAK";
  }
}

const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  console.log(countScore(line))
})




// HJ88 扑克牌大小
var readLine = require("readline");
rl = readLine.createInterface({
  input: process.stdin,
  output: process.stdout,
});

rl.on("line", function (line) {
  const dic = {
    '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
    '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12,
    '2': 13, 'joker': 14, 'JOKER': 15
  }
  let t;
  let s1, s2
  let lst1 = lst2 = []
  t = line.trim().split('-')
  s1 = t[0]
  s2 = t[1]
  lst1 = s1.split(' ');
  lst2 = s2.split(' ');
  [L1 = 0, L2 = 0] = [lst1.length, lst2.length]
  if (L1 === L2) {
    if (dic[lst1[0]] > dic[lst2[0]]) {
      console.log(s1)
    } else {
      console.log(s2)
    }
  } else {
    if (s1 === 'joker JOKER' || s2 === 'joker JOKER') {
      console.log('joker JOKER')
    } else if (isboom(lst1)) {
      console.log(s1)
    } else if (isboom(lst2)) {
      console.log(s2)
    } else {
      console.log('ERROR')
    }
  }
})
function isboom(lst) {
  if (lst.length == 4 && (new Set(lst)).size === 1) {
    return true
  }
  return false
}




// HJ89 24点运算
var readLine = require("readline");
rl = readLine.createInterface({
  input: process.stdin,
  output: process.stdout,
});

var lineNum = 1;
var inputs = [];
rl.on("line", function (input) {
  inputs.push(input.trim());
  if (inputs.length === lineNum) {
    algorithmExe(inputs);
    inputs = [];
  }
});

function puke2Num(str) {
  var num = Number(str);
  if (!isNaN(Number(str))) {
    return num;
  }
  if (str === "J") return 11;
  if (str === "Q") return 12;
  if (str === "K") return 13;
  if (str === "A") return 1;
  return null;
}
function check(nums) {
  for (let i = 0; i < nums.length; i++) {
    const element = nums[i];
    if (!element) {
      return false;
    }
  }
  return true;
}

function extracOpe(ope) {
  switch (ope) {
    case 0:
      return "+";
    case 1:
      return "-";
    case 2:
      return "*";
    case 3:
      return "/";
  }
}

function calc(a, b) {
  return [a + b, a - b, a * b, a / b];
}
function algorithmExe(inputArr) {
  var pukeNums = inputArr[0].split(" ").map((item) => puke2Num(item));
  if (!check(pukeNums)) {
    console.log("ERROR");
    return;
  }
  for (let i = 0; i < pukeNums.length; i++) {
    const a = pukeNums[i];
    for (let j = 0; j < pukeNums.length; j++) {
      if (j === i) continue;
      const b = pukeNums[j];
      var calcs = calc(a, b);
      for (let k = 0; k < calcs.length; k++) {
        var ab = calcs[k];
        for (let m = 0; m < pukeNums.length; m++) {
          if (m !== i && m !== j) {
            var calcs1 = calc(ab, pukeNums[m]);
            for (let l = 0; l < calcs1.length; l++) {
              const abc = calcs1[l];
              for (let n = 0; n < pukeNums.length; n++) {
                if (n !== m && n !== j && n !== i) {
                  var abcd = calc(abc, pukeNums[n]);
                  for (let o = 0; o < abcd.length; o++) {
                    if (abcd[o] === 24) {
                      logResult(
                        pukeNums[i],
                        pukeNums[j],
                        pukeNums[m],
                        pukeNums[n],
                        k,
                        l,
                        o
                      );
                      return;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  console.log("NONE");
}

function num2Puke(num) {
  if (num >= 2 && num <= 10) {
    return num;
  }
  if (num === 11) return "J";
  if (num === 12) return "Q";
  if (num === 13) return "K";
  if (num === 1) return "A";
}

function logResult(a1, b1, c1, d1, ope1, ope2, ope3) {
  var a = num2Puke(a1);
  var b = num2Puke(b1);
  var c = num2Puke(c1);
  var d = num2Puke(d1);
  var str = a + extracOpe(ope1) + b + extracOpe(ope2) + c + extracOpe(ope3) + d;
  console.log(str);
}





// HJ90 合法IP
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})
rl.on('line', function (line) {
  const ipArr = line.split('.')
  if (ipArr.length !== 4) {
    console.log('NO')
    return
  }
  for (let i = 0; i < ipArr.length; i++) {
    const ele = ipArr[i];
    const notNum = ele.replace(/[0-9]/g, '')
    const num = Number(ele)
    // 判断当前ip是否有值 及 将数字都替换为空，如果还存在字符，则说明不是存在额外的字符串 以及判断ip地址的范围是否越界
    if ((ele == null) || (ele == '') || notNum.length > 0 || num > 255 || num < 0) {
      console.log('NO')
      return
    }
    if (ele.length > 1 && ele[0] == '0') {
      console.log('NO')
      return
    }
  }
  console.log('YES')
})




// HJ91 走方格的方案数
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

void async function () {
  // Write your code here
  while (line = await readline()) {
    let [m, n] = line.split(' ').map(Number);
    let dp = Array(m + 1).fill(0).map(() => new Array(n + 1).fill(1));
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
      }
    }
    console.log(dp[m][n])
  }
}()



// HJ92 在字符串中找出连续最长的数字串
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
rl.on("line", function (line) {
  let re = /\d+/g
  let list = line.match(re)
  let oldList = line.match(re)
  //降序
  list.sort((a, b) => {
    return parseInt(b) - parseInt(a)
  })
  let maxLength = list[0].length
  let maxStr = []
  oldList.forEach(c => {
    if (c.length == maxLength) {
      maxStr.push(c)
    }
  })
  console.log(maxStr.join("") + "," + maxLength)
});




// HJ93 数组分组
//递归，others第三组有两种情况，根本没有值，无须分配； 有值，需要全部分配完
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
let k = 0;
let n = 0;
let nums = [];
rl.on('line', function (line) {
  if (k === 0) {
    n = Number(line);
    k++;
  } else {
    nums = line.split(' ').map(e => Number(e));
    getArr();
  }
});

function getArr() {
  let multi5 = [];
  let multi3 = [];
  let others = [];
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] % 3 === 0 && nums[i] % 5 !== 0) {
      multi3.push(nums[i]);
    } else if (nums[i] % 5 === 0) {
      multi5.push(nums[i])
    } else {
      others.push(nums[i]);
    }
  }
  let sum5 = multi5.length > 0 ? multi5.reduce((prev, current) => prev + current) : 0;
  let sum3 = multi3.length > 0 ? multi3.reduce((prev, current) => prev + current) : 0;
  console.log(isExists(sum5, sum3, others, 0));
}

function isExists(sum5, sum3, others, index) {
  if (others.length === 0 && sum5 !== sum3) {
    return false;
  } else if (others.length === 0 && sum5 === sum3) {
    return true;
  } else if (others.length === index && sum5 !== sum3) {
    return false;
  } else if (others.length === index && sum5 === sum3) {
    return true;
  } else if (index < others.length) {
    return isExists(sum5 + others[index], sum3, others, index + 1) || isExists(sum5, sum3 + others[index], others, index + 1);
  } else {
    return false;
  }
}




// HJ94 记票统计
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

void async function () {

  let num1 = await readline();
  let name = (await readline()).split(' ');
  let num2 = await readline();
  let res = (await readline()).split(' ');

  const map = new Map();
  for (let i = 0; i < name.length; i++) {
    map.set(name[i], 0)
  }
  map.set('Invalid', 0)

  for (let j = 0; j < res.length; j++) {
    if (map.has(res[j])) {
      map.set(res[j], map.get(res[j]) + 1)
    } else {
      map.set('Invalid', map.get('Invalid') + 1)
    }
  }
  for (const [key, value] of map) {
    console.log(`${key} : ${value}`)
  }
}()




// HJ95 人民币转换
let num = "151121.15"
// let num = "1010.00"
// let num = "6007.14"

const unit = ['元', '整', '角', '分'];
const unitMin = ['', '拾', '佰', '仟'], unitMax = ['', '万', '亿', '万亿'];
const arr = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖'];

let [integer, decimal] = num.split('.');
let len = Math.ceil(integer.length / 4);
let val = "人民币", arrInt = [];//每四个一组，从后向前截取 

for (let i = 1; i <= len; i++) {
  const startEnd = i == 1 ? [-i * 4] : [-i * 4, (-i + 1) * 4]
  arrInt.push(integer.slice(...startEnd));
}
for (let i = len - 1; i >= 0; i--) {
  let w = readFour(arrInt[i]);
  if (w) {
    val += w + unitMax[i];
  }
}
val += unit[0]

//小数位读取
if (decimal == "00") {
  val += unit[1]
} else {
  val += arr[decimal[0]] + (decimal[0] > 0 ? unit[2] : '')
  if (decimal[1] > 0) {
    val += arr[decimal[1]] + unit[3]
  }
}
//最终读取结果
console.log(val)

//每四位读取
function readFour(item) {
  let r = '', len = item.length;
  for (let i = len - 1, j = 0; i >= 0; i--, j++) {
    //对壹拾去壹，并去掉重复的零       
    let n = ((i == 1 && item[j] == 1) ? "" : arr[item[j]]) + (item[j] > 0 ? unitMin[i] : '');
    let prev = r.slice(-1);
    if (!(prev == arr[0] && n == arr[0])) {
      r += n
    }
  }
  r = r.replace(/['零']$/, "");
  return r
}



// HJ96 表示数字
\const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  console.log(line.replace(/[0-9]+/g, (val) => '*' + val + '*'))
})



// HJ97 记负均正
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
let inputs = []
rl.on('line', (line) => {
  inputs.push(line);
})
rl.on('close', () => {
  const arr = inputs.slice(1)[0].split(' ');
  let count = 0, count2 = 0, other = 0;
  let res
  arr.forEach(i => {
    if (i < 0) {
      count += 1;
    } else if (i > 0) {
      count2 += 1;
      other += Number(i);
    }
  })
  if (count2) {
    res = (other / count2).toFixed(1)
  } else {
    res = '0.0';
  }
  console.log(count + ' ' + res)
})




// HJ98 自动售货系统
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

void async function () {
  // Write your code here
  while (line = await readline()) {
    // console.log(line);
    let inputs = line.split(';')
    let things = {
      'A1': {
        price: 2,
        num: 0
      },
      'A2': {
        price: 3,
        num: 0
      },
      'A3': {
        price: 4,
        num: 0
      },
      'A4': {
        price: 5,
        num: 0
      },
      'A5': {
        price: 8,
        num: 0
      },
      'A6': {
        price: 6,
        num: 0
      },
    }
    let money = {
      1: 0,
      2: 0,
      5: 0,
      10: 0,
    }
    let yue = 0
    for (let input of inputs) {
      let ins = input.split(' ')
      if (ins[0] === 'r') {
        Object.keys(things).map((item, index) => {
          things[item].num = ins[1].split('-')[index] * 1
        })
        Object.keys(money).map((item, index) => {
          money[item] = ins[2].split('-')[index] * 1
        })
        console.log('S001:Initialization is successful')
      }
      if (ins[0] === 'p') {
        if (['1', '2', '5', '10'].includes(ins[1])) {
          if (ins[1] * 1 > 2 && ins[1] * 1 > money[1] + money[2] * 2) {
            console.log('E003:Change is not enough, pay fail')
          } else if (soldOut(things)) {
            console.log('E005:All the goods sold out')
          } else {
            money[ins[1]] += 1
            yue += ins[1] * 1
            console.log(`S002:Pay success,balance=${yue}`)
          }
        } else {
          console.log('E002:Denomination error')
        }
      }
      if (ins[0] === 'b') {
        if (things[ins[1]]) {
          if (things[ins[1]].num === 0) {
            console.log('E007:The goods sold out')
          } else if (things[ins[1]].price > yue) {
            console.log('E008:Lack of balance')
          } else {
            yue -= things[ins[1]].price
            console.log(`S003:Buy success,balance=${yue}`)
          }

        } else {
          console.log('E006:Goods does not exist')
        }
      }
      if (input === 'c') {
        if (yue === 0) {
          console.log('E009:Work failure')
        } else {
          c(yue, money)
          yue = 0
        }
      }
      if (input[0] === 'q') {
        if (ins[0] === 'q') {
          if (ins[1] === '0') {
            Object.keys(things).map(item => {
              console.log(`${item} ${things[item].price} ${things[item].num}`)
            })
          } else if (ins[1] === '1') {
            Object.keys(money).map(item => {
              console.log(`${item} yuan coin number=${money[item]}`)
            })
          } else {
            console.log('E010:Parameter error')
          }
        } else {
          console.log('E010:Parameter error')
        }
      }
    }
  }
}()

function soldOut(things) {
  let bool = true
  Object.keys(things).map(item => {
    if (things[item].num !== 0) {
      bool = false
    }
  })
  return bool
}
function c(yue, money) {
  let s = yue
  let list = {
    1: 0,
    2: 0,
    5: 0,
    10: 0,
  }
  let keys = Object.keys(money).reverse(), index = 0
  while (s > 0 && index < keys.length) {
    if (s >= keys[index]) {
      if (money[keys[index]]) {
        s -= keys[index]
        list[keys[index]] = list[keys[index]] + 1
        money[keys[index]] = money[keys[index]] - 1
      } else {
        index++
      }

    } else {
      index++
    }
  }
  Object.keys(list).map(item => {
    console.log(`${item} yuan coin number=${list[item]}`)
  })
}





// HJ99 自守数
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  let res = 0;
  for (let i = 0; i <= line; i++) {
    const n = String(i).length
    if (String(i * i).slice(-n) === String(i)) {
      res++;
    }
  }
  console.log(res)
})




// HJ100 等差数列
const count = (n) => {
  return n * 2 + n * (n - 1) * 3 / 2;
}
/**
算法分析：
很简单的一道高中数学题：等差数列求前N项和。
复习一下，公式为：Sn = n * a1 + n * ( n - 1 ) * d / 2 （n为前多少项；a1为首项；d为公差）*/
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  console.log(count(line))
})




// 
HJ101 输入整型数组和排序标识，对其元素按照升序或降序进行排序
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
let totle = 0;
let arr = [];
let sort = 0;
rl.on("line", function (line) {
  totle++;
  if (totle === 2) {
    arr = line.split(" ").map(item => Number(item))
  } else if (totle === 3) {
    !Boolean(Number(line)) ? console.log(arr.sort((x, y) => x - y).join(" ")) : console.log(arr.sort((x, y) => y - x).join(" "))
  }
});




// HJ102 字符统计
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', (line) => {
  let obj = {};
  for (let i of line) {
    if (obj[i]) {
      obj[i]++;
    } else {
      obj[i] = 1
    }
  }
  const arr = Array.from(new Set(line.split('')));// 去重
  arr.sort((a, b) => {
    if (obj[a] === obj[b]) {
      return a.charCodeAt(0) - b.charCodeAt(0)//升序
    } else {
      return obj[b] - obj[a]//降序
    }
  })

  console.log(arr.join(''))
})



// HJ103 Redraiment的走法
while (readline()) {
  let arr = readline().split(" ").map(Number);
  let len = arr.length;
  // dp[i] 表示以第 i 个桩为结尾，最多走多少步，初始是 1 步（默认这个桩是跟它之前相比最矮的）
  let dp = new Array(len).fill(1);
  for (let i = 0; i < len; i++) {
    for (let j = 0; j < i; j++) {
      if (arr[i] > arr[j]) {
        dp[i] = Math.max(dp[i], dp[j] + 1);
        // 如果第i个桩前面有比它矮的（比如是j），
        // 且以第j个桩为结尾走的步数是最多的，
        // 步数就是dp[j]+1，加的这个1表示从第j个走1步到第i个桩；另一种就是dp[i],默认等于1，但是
        // 遍历j的过程可能会更新这个值，因此取上述两个结果中最大的那个值，表示第i个桩为结尾，
        // 最多走多少步
      }
    }
  }
  print(Math.max(...dp));
}




// HJ105 记负均正II
const rl = require("readline").createInterface({ input: process.stdin });
var iter = rl[Symbol.asyncIterator]();
const readline = async () => (await iter.next()).value;

void async function () {
  // Write your code here
  let count = 0;
  let tuple = {};
  tuple[0] = 0;
  tuple[1] = 0;
  while (line = await readline()) {
    let n = parseInt(line);
    if (n < 0) {
      tuple[0]++;
    } else {
      tuple[1] += n;
      count++;
    }
  }
  console.log(tuple[0]);
  console.log(count > 0 ? (tuple[1] / count).toFixed(1) : Number(0).toFixed(1));
}()




// HJ106 字符逆序
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', function (line) {
  console.log(line.split('').reverse().join(''))
});




// HJ107 求解立方根
let x = 0.0001;
let low = Math.min(-1.0, a);
let high = Math.max(1.0, a);
let ans = (low + high) / 2; //设置中间值
while (Math.abs(ans ** 3 - a) >= x) {
  if (ans ** 3 < a) {
    low = ans; //向右找
  } else {
    high = ans; //向左找
  }
  ans = (low + high) / 2;
}
console.log(ans.toFixed(1));



// HJ108 求最小公倍数
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', function (line) {
  let [a, b] = line.split(' ');
  function gcd(a, b) {
    if (a % b == 0) return b;
    return gcd(b, a % b);
  }

  console.log(a * b / gcd(a, b));
});

// 输入n个互不相同的二维整数坐标，这n个坐标可以构成的正方形数量

// 张兵和王武是五子棋迷，工作之余经常切磋棋艺。这步，这会儿游戏起来了。走了一会儿轮张兵了，对着一条线思考起来了，这条线上的妻子分布如下：
// 用数组表示：-1 0 1 1 1 0 1 0 1 - 1
// 妻子分布说明：
// 1. - 1代表白字，0代表空位，1代表黑子
// 2.数组长度L，满足1 < L < 40，且L为技术

// 你得帮他写一个程序，算出最有利的出资位置。最有利意义：

// 跳房子，也叫做跳飞机，是一种世界性儿童游戏。you游戏参与者需要分多个回合顺序调到第1个知道房子的最后一格，然后活的一次选房子的机会，知道所有房子被选完，房子最多的人获胜。
