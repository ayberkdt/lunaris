const THREE = require('three');

const dummy = new THREE.Object3D();
const target = new THREE.Vector3(-10, 0, 0); // Moving left
dummy.position.set(0, 0, 0);
dummy.lookAt(target);

console.log("After lookAt:");
const zAxis = new THREE.Vector3(0, 0, 1).applyQuaternion(dummy.quaternion);
const minusZAxis = new THREE.Vector3(0, 0, -1).applyQuaternion(dummy.quaternion);
console.log("+Z axis points to:", zAxis);
console.log("-Z axis points to:", minusZAxis);

dummy.rotateY(Math.PI);

console.log("After rotateY(PI):");
const newZAxis = new THREE.Vector3(0, 0, 1).applyQuaternion(dummy.quaternion);
const newMinusZAxis = new THREE.Vector3(0, 0, -1).applyQuaternion(dummy.quaternion);
console.log("+Z axis points to:", newZAxis);
console.log("-Z axis points to:", newMinusZAxis);
