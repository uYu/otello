import {spawnSync} from 'node:child_process';
for(const [command,args] of [['python3',['scripts/prepare-web-book.py']],[process.execPath,['scripts/check-web-release.mjs','--write']]]){
 const result=spawnSync(command,args,{stdio:'inherit'});if(result.error)throw result.error;if(result.status!==0)process.exit(result.status??1);
}
