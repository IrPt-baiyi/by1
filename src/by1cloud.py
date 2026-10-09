#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1cloud -- **驱动一台租来的显卡机器**（AutoDL 或任何 SSH 可达的）。

## 我能做和不能做的

    ✗ 我不能    注册账号、付费、拿到机器
    ✓ 我能      拿到 host + port + 一把密钥之后，**把整套验证跑完**

所以这个脚本做的是后半段。前面那一段要你做：

    1. 在 AutoDL 租一台（选 PyTorch 镜像，省掉装 torch 那一步）
    2. 把下面这把公钥贴进控制台的「SSH 公钥」：

         ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHvjiAl9GQE7KY7bom2ihju40mKOHzO92xvvncUWUIeL by1-autodl

    3. 把它给的 `ssh -p <端口> root@<区域>.autodl.com` 里的
       host 和 port 告诉我

## 为什么必须用密钥，不能用密码

这个脚本从非交互的 shell 里调 `ssh` —— **没有地方输密码**。
AutoDL 支持上传公钥，所以走密钥这条路是通的，密码那条不通。

## 用法

    python by1cloud.py --host region.autodl.com --port 12345 check
    python by1cloud.py --host ... --port ... push      # 传包上去
    python by1cloud.py --host ... --port ... setup     # 装依赖
    python by1cloud.py --host ... --port ... run       # 跑验证
    python by1cloud.py --host ... --port ... all       # 一条龙
"""
import argparse
import os
import subprocess
import sys
import tarfile
import time
import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.expanduser('~/.ssh/by1_autodl')
REMOTE = '/root/by1'
PKG = os.path.join(os.path.dirname(HERE), 'by1-cloud.tar.gz')

# **SSH 的公共参数。** 三个都不能省：
#   BatchMode       没有地方输密码，失败要立刻失败，不要挂在那儿等
#   StrictHostKey   第一次连的机器没进 known_hosts，非交互会卡住
#   ConnectTimeout  连不上要报错，不要等
# **空设备：两个平台写法不同。** 见 SSH_COMMON 里的注释。
NULL_DEV = 'NUL' if os.name == 'nt' else '/dev/null'

SSH_COMMON = ['-o', 'BatchMode=yes',
              '-o', 'StrictHostKeyChecking=no',
              # **空设备要在两边都对。** 这个脚本从 Windows 上跑，
              # 而 /dev/null 是 Linux 的写法 —— Windows 的 ssh 认不出来。
              '-o', 'UserKnownHostsFile=' + NULL_DEV,
              '-o', 'ConnectTimeout=15',
              '-o', 'LogLevel=ERROR']

# **依赖要钉版本，而且钉的是 transformers。**
#
# by1 自己的代码对 torch 几乎没要求（rsqrt / tril / atan2 / finfo
# 全是老 API），**任何 torch >= 2.0 都行**。
#
# 真正的约束在 transformers：判卷人要 GptOssConfig / Qwen3NextConfig，
# 而**5.x 改过因果掩码的行为** —— HF 在 `attention_mask=None` 时
# 不再自动做因果，这个坑在 MLA 和 GPT-2 上各踩过一次。
#
# 所以：**判卷人换了就不是同一个判卷人。**
# 版本不一致的话，改的可能是参考，不是 by1。
DEPS = 'numpy transformers==5.15.1 safetensors'

# **镜像里 `python` 不在 PATH 上。** AutoDL 用 conda，
# 解释器在 /root/miniconda3/bin/。所有远程命令都用这个全路径。
PY = '/root/miniconda3/bin/python'


def ssh_password(host, port, user, password, cmd, timeout=600):
    """**用密码连一次 —— 只为了把公钥装进去。**

    ## 为什么不能一直用密码

    这个脚本从非交互的 shell 里调 `ssh` —— **没有 TTY，密码没地方输**。
    `sshpass` 这台机器上也没有。所以密码只能用来做一件事：

        把公钥写进 ~/.ssh/authorized_keys

    之后所有命令都走密钥。**密码只用一次。**

    ## 用完请改密码

    密码会进会话记录。实例是临时的、用完就释放，所以风险有界 ——
    但改一下更省心。

    用 paramiko 而不是 `ssh` 是因为：**它能在代码里收密码**，
    而 `ssh` 只会去读 TTY。
    """
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=password,
              timeout=20, allow_agent=False, look_for_keys=False)
    _in, out, err = c.exec_command(cmd, timeout=timeout)
    o = out.read().decode('utf-8', 'replace')
    e = err.read().decode('utf-8', 'replace')
    rc = out.channel.recv_exit_status()
    c.close()
    return rc, o + e


def bootstrap_key(args):
    """把公钥装到远端 —— **密码唯一该做的事。**"""
    pub = args.key + '.pub'
    if not os.path.exists(pub):
        print('  [FAIL] 没有公钥 %s —— 先跑 python by1cloud.py key' % pub)
        return 1
    pubtext = by1io.read_text(pub, encoding='utf-8').strip()
    print('  ── 用密码连一次，把公钥装进去 ──')
    cmd = ('mkdir -p ~/.ssh && chmod 700 ~/.ssh && '
           'grep -qF "%s" ~/.ssh/authorized_keys 2>/dev/null || '
           'echo "%s" >> ~/.ssh/authorized_keys; '
           'chmod 600 ~/.ssh/authorized_keys; echo INSTALLED; '
           'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
           % (pubtext, pubtext))
    try:
        rc, out = ssh_password(args.host, args.port, args.user,
                               args.password, cmd)
    except Exception as e:
        print('  [FAIL] 密码登录失败：%s' % str(e)[:90])
        return 1
    for line in out.splitlines():
        print('    ' + line)
    if 'INSTALLED' not in out:
        print('  [FAIL] 公钥没装进去')
        return 1
    print()
    print('  [OK] 公钥装好了 —— **以后不用密码了**')
    # 立刻用密钥验一次，确认真的通了
    print('  ── 用密钥验一次 ──')
    return cmd_check(args)


def ssh(args, cmd, timeout=600, quiet=False):
    """跑一条远程命令。**走 paramiko，不走 OpenSSH。**

    ## 为什么不用系统的 ssh

    实测过：AutoDL 那个网关（`connect.*.seetacloud.com`）的 sshd
    **不认 `publickey-hostbound-v00@openssh.com`** —— 那是 OpenSSH 8.2+
    的客户端扩展。现象很误导：

        debug1: Server accepts key: ...        <- 服务端说接受
        debug3: sign_and_send_pubkey: using publickey-hostbound-v00
        Permission denied (publickey,password) <- 然后拒了

    公钥在 authorized_keys 里、权限 700/600、`pubkeyauthentication yes`
    —— **全都对**，但就是不通。

    paramiko 不发那个扩展，所以它通。**这里不是图省事，是 OpenSSH
    那条路走不通。**

    顺带：镜像里 `python` 不在 PATH 上，要 `/root/miniconda3/bin/python`。
    """
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(hostname=args.host, port=args.port, username=args.user,
              timeout=25, allow_agent=False, look_for_keys=False)
    # 密钥有口令就用口令；没口令就只用密钥。
    if getattr(args, 'password', None):
        kw['password'] = args.password
    if os.path.exists(args.key):
        kw['key_filename'] = args.key
    # **重试** —— 网关偶发拒绝连接（"连太快了"）。
    import time as _t
    last = None
    for attempt in range(4):
        try:
            c.connect(**kw)
            last = None
            break
        except Exception as ex:
            last = ex
            _t.sleep(3 + attempt * 4)
    if last is not None:
        raise last
    # **PATH 要自己补。**
    # 非交互 SSH 的 PATH 是极简的 —— 于是 python / pip / gcc
    # **全都找不到**。一个原因，三个症状，我一开始当成三件事查。
    _PATH = ('export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:'
             '/usr/bin:/sbin:/bin:/root/miniconda3/bin:; ')
    _i, o, e = c.exec_command(_PATH + cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace')
    err = e.read().decode('utf-8', 'replace')
    rc = o.channel.recv_exit_status()
    c.close()
    if not quiet:
        for line in (out + err).splitlines():
            print('    ' + line)
    return rc, out + err


def scp_up(args, local, remote):
    """传文件。**同样走 paramiko**（SFTP），理由和上面一样。"""
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(hostname=args.host, port=args.port, username=args.user,
              timeout=25, allow_agent=False, look_for_keys=False)
    if getattr(args, 'password', None):
        kw['password'] = args.password
    if os.path.exists(args.key):
        kw['key_filename'] = args.key
    c.connect(**kw)
    sftp = c.open_sftp()
    try:
        sftp.put(local, remote)
        rc = 0
    except Exception as e:
        print('    !! SFTP 失败：%s' % str(e)[:80])
        rc = 1
    sftp.close()
    c.close()
    return rc


def cmd_check(args):
    print('  ── 连得上吗，卡是什么 ──')
    rc, out = ssh(args, 'nvidia-smi --query-gpu=name,memory.total,'
                        'driver_version,compute_cap --format=csv,noheader; '
                        '%s -c "import torch;print(\'torch\','
                        'torch.__version__,\'cuda\',torch.version.cuda,'
                        '\'avail\',torch.cuda.is_available())" 2>&1 | tail -1; '
                        'which gcc || echo "没有 gcc"; '
                        'df -h / | tail -1' % PY)
    if rc != 0:
        print()
        print('  [FAIL] 连不上（退出码 %d）' % rc)
        print('  检查：host / port / 公钥有没有贴进控制台 / 实例开没开')
        return 1
    print()
    print('  [OK] 连上了')
    return 0


def cmd_push(args):
    print('  ── 打包并传上去 ──')
    # 只用 git 里跟踪的文件 —— 把 25MB 的 refs/ 也传上去没意义
    r = subprocess.run(['git', 'ls-files'], capture_output=True, text=True,
                       cwd=by1paths.ROOT)
    # **git 失败时不能照常打包。** 退出码非零的话 stdout 可能是空的，
    # 于是打出一个**几乎空**的包、照样传上去 —— 而远端要到
    # `tar xzf` 之后才发现。这里直接停，比传一个坏包便宜。
    if r.returncode != 0:
        print('  [FAIL] `git ls-files` 退出码 %d：%s'
              % (r.returncode, (r.stderr or '').strip()[:120]))
        print('         打包要靠它列文件 —— 没有清单就不打。')
        return 1
    files = [f for f in r.stdout.split() if os.path.exists(os.path.join(by1paths.ROOT, f))]
    if not files:
        print('  [FAIL] `git ls-files` 返回 0 个文件 —— 不做空包')
        return 1
    with tarfile.open(PKG, 'w:gz') as tf:
        for f in files:
            tf.add(os.path.join(by1paths.ROOT, f), arcname=f)
    print('    %d 个文件，%.1f MB' % (len(files), os.path.getsize(PKG) / 1e6))
    ssh(args, 'mkdir -p ' + REMOTE, quiet=True)
    rc = scp_up(args, PKG, REMOTE + '/pkg.tar.gz')
    if rc != 0:
        print('  [FAIL] scp 失败')
        return 1
    ssh(args, 'cd %s && tar xzf pkg.tar.gz && ls | head -5' % REMOTE)
    print('  [OK] 传上去了')
    return 0


def cmd_setup(args):
    print('  ── 装依赖 ──')

    # **C 后端要 gcc。** Ubuntu 22.04 的镜像一般有，但精简镜像没有。
    # 少它的话，`by1all` 会「找不到 gcc」而整段 C 的检查都跳过 ——
    # 那不是失败，是**静默少了一半验证**。
    rc, out = ssh(args, 'which gcc || echo NOGCC', quiet=True)
    if 'NOGCC' in out or 'no gcc' in out.lower():
        print('    没有 gcc —— 装一个（C 后端要它）')
        ssh(args, 'apt-get update -qq && apt-get install -y -qq build-essential '
                  '2>&1 | tail -2', timeout=900)
    else:
        print('    gcc %s' % out.strip().splitlines()[-1][:60])

    # AutoDL 的 PyTorch 镜像一般自带 torch+cu，所以先看有没有
    rc, out = ssh(args, '%s -c "import torch;print(torch.cuda.is_available())"'
                        ' 2>&1 || echo MISSING' % PY, quiet=True)
    if 'True' in out:
        print('    镜像自带 torch 且认得出显卡 —— 只补判卷人要的')
        ssh(args, '%s -m pip install -q %s 2>&1 | tail -2' % (PY, DEPS))
    else:
        print('    没有可用的 torch+cu —— 装一套（这一步慢，几分钟）')
        ssh(args, '%s -m pip install -q torch --index-url ' % PY +
                  'https://download.pytorch.org/whl/cu124 2>&1 | tail -2',
            timeout=2400)
        ssh(args, '%s -m pip install -q %s 2>&1 | tail -2' % (PY, DEPS), timeout=1200)
    print('  [OK]')
    return 0


def cmd_run(args):
    """跑验证。

    ## 为什么合成**一条**远程命令

    实测：AutoDL 的网关**限制连接频率**。一步一步各开一个连接的话，
    第二条就报 `Error reading SSH protocol banner` ——
    而那个报错长得像网络问题，其实是"你连太快了"。

    合起来也更快：开一次连接的开销省掉三次。
    """
    print('  ── 逐层跑验证 ──')
    steps = [
        ('① 设备无关性（本地也能跑，这里再跑一遍）',
         '%s by1dev.py 2>&1 | tail -4' % PY),
        ('② **显卡**：搬得过去吗、两边数一致吗',
         '%s by1gpu.py 2>&1 | tail -30' % PY),
        ('③ 端到端：真产物 -> IR -> 三后端 -> 官方实现',
         '%s by1e2e.py 2>&1 | tail -14' % PY),
        ('④ 全量（含 C 后端）',
         '%s by1all.py 2>&1 | tail -10' % PY),
    ]
    # **一条命令里串起来，中间加分隔符** —— 这样输出还能分辨是哪一步。
    script = ' ; '.join(
        'echo "===STEP===%s" ; cd %s && (%s) ; echo "===RC=$?==="'
        % (label.replace('"', ''), REMOTE, cmd)
        for label, cmd in steps)
    rc, out = ssh(args, script, timeout=7200)
    print()
    print('  ── 汇总 ──')
    for line in out.splitlines():
        if line.startswith('===STEP==='):
            print()
            print('  ── ' + line.replace('===STEP===', ''))
    bad = out.count('[FAIL]') + out.count('项失败')
    print()
    print('  输出里出现 [FAIL] / 失败 %d 处' % bad)
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description='驱动一台租来的显卡机器')
    ap.add_argument('action',
                    choices=['check', 'push', 'setup', 'run', 'all', 'key',
                             'bootstrap'])
    ap.add_argument('--host')
    ap.add_argument('--port', type=int)
    ap.add_argument('--user', default='root')
    ap.add_argument('--key', default=KEY)
    ap.add_argument('--password', help='**只用一次**，装公钥用的。'
                                       '装完就改掉它。')
    args = ap.parse_args()

    if args.action == 'key':
        pub = args.key + '.pub'
        if os.path.exists(pub):
            print(by1io.read_text(pub, encoding='utf-8').strip())
            print()
            print('  贴到 AutoDL 控制台的「SSH 公钥」里。')
            return 0
        print('  没有公钥 —— 先生成： ssh-keygen -t ed25519 -f %s -N ""' % args.key)
        return 1

    if args.action == 'bootstrap':
        if not (args.host and args.port and args.password):
            print('  bootstrap 要 --host / --port / --password')
            return 2
        return bootstrap_key(args)

    if not args.host or not args.port:
        print('  要 --host 和 --port。')
        print('  AutoDL 控制台上写的是： ssh -p <port> root@<host>')
        print('  没租过机器的话： python by1cloud.py key  （先拿公钥）')
        return 2
    if not os.path.exists(args.key):
        print('  找不到密钥 %s —— 先跑 python by1cloud.py key' % args.key)
        return 2

    t0 = time.time()
    if args.action == 'all':
        # **给了密码就先装公钥** —— 之后全走密钥，密码只用这一次。
        if args.password:
            if bootstrap_key(args) != 0:
                return 1
        for step in (cmd_check, cmd_push, cmd_setup, cmd_run):
            rc = step(args)
            if rc != 0:
                return rc
    else:
        rc = {'check': cmd_check, 'push': cmd_push,
              'setup': cmd_setup, 'run': cmd_run}[args.action](args)
        if rc != 0:
            return rc
    print()
    print('  共用 %.0f 秒' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
