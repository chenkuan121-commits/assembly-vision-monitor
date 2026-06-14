import os
import json
from datetime import datetime

class NGProductTracker:
    def __init__(self):
        # 核心数据结构重构：按方案名称隔离数据
        # 格式: { "方案A": {"ok_count": 0, "ng_count": 0, "history": []}, ... }
        self.profiles_data = {}
        self.active_profile = "默认方案"
        self.current_product = None
        self.temp_total_count = 0  # 仅用于生成递增的产品 ID

    def _get_current_db(self):
        """获取当前激活方案的数据集，如果没有则自动创建"""
        if self.active_profile not in self.profiles_data:
            self.profiles_data[self.active_profile] = {"ok_count": 0, "ng_count": 0, "history": []}
        return self.profiles_data[self.active_profile]

    def switch_profile(self, profile_name):
        """切换工序方案时，如果有没做完的产品，直接丢弃（不计入数据）"""
        if self.current_product:
            self.current_product = None  
        self.active_profile = profile_name

    # ── 产品级操作 ──
    def start_product(self, process_steps):
        """新一个产品开始装配"""
        if self.current_product:
            self._finalize_product()
            
        self.temp_total_count += 1
        now = datetime.now()
        self.current_product = {
            'id': self.temp_total_count,
            'date': now.strftime("%Y-%m-%d"),
            'start_time': now.strftime("%H:%M:%S"),
            'started_at': now.isoformat(timespec='seconds'),
            'end_time': '',
            'status': 'OK',
            'total_steps': len(process_steps),
            'step_records': [],
            'jump_alarms': [],
        }
        for i, s in enumerate(process_steps):
            text = s.get('text', '') if isinstance(s, dict) else str(s)
            self.current_product['step_records'].append({
                'step': i + 1,
                'text': text[:60],
                'status': 'pending',
            })

    def _finalize_product(self, unfinished_reason='产品结束时未完成', ng_reason='工序未完整走完'):
        """结束时把产品归档到当前方案的历史记录中"""
        if not self.current_product:
            return
            
        db = self._get_current_db()
        
        # 检查是否有未完成的步骤
        for rec in self.current_product['step_records']:
            if rec['status'] == 'pending':
                rec['status'] = 'skipped'
                rec['reason'] = unfinished_reason
                self._mark_ng(ng_reason)
                
        self.current_product['end_time'] = datetime.now().strftime("%H:%M:%S")
        
        # 结算数量
        if self.current_product['status'] == 'OK':
            db["ok_count"] += 1
        else:
            db["ng_count"] += 1
            
        # 存入历史：NG 一定保存；曾触发告警但已补救/AOI 恢复的产品也保存，便于追溯
        if self.current_product['status'] == 'NG' or self._has_audit_event(self.current_product):
            db["history"].append(self.current_product)
            if len(db["history"]) > 200:
                db["history"] = db["history"][-200:]
                
        self.current_product = None

    def _mark_ng(self, reason=''):
        if self.current_product:
            self.current_product['status'] = 'NG'
            if reason:
                self.current_product['ng_reason'] = reason

    def finalize_as_ng(self, reason):
        if not self.current_product:
            return
        self._mark_ng(reason)
        self._finalize_product(unfinished_reason=reason, ng_reason=reason)

    def _has_audit_event(self, product):
        if product.get('jump_alarms'):
            return True
        for rec in product.get('step_records', []):
            if rec.get('status') in ('skipped', 'remedied'):
                return True
            if any(rec.get(k) for k in (
                'aoi_blocked', 'aoi_recovered', 'aoi_forced',
                'timeout_warning', 'was_aoi_blocked'
            )):
                return True
        return False

    def has_product_activity(self, min_elapsed_sec=0):
        if not self.current_product:
            return False
        if self.current_product.get('status') == 'NG':
            return True
        if self._has_audit_event(self.current_product):
            return True
        if any(rec.get('status') != 'pending' for rec in self.current_product.get('step_records', [])):
            return True
        if min_elapsed_sec > 0:
            try:
                started_at = datetime.fromisoformat(self.current_product.get('started_at', ''))
                return (datetime.now() - started_at).total_seconds() >= min_elapsed_sec
            except Exception:
                return False
        return False

    # ── 步骤级操作 ──
    def mark_step_completed(self, step_idx, step_text=''):
        cp = self.current_product
        if cp and step_idx < len(cp['step_records']):
            current_status = cp['step_records'][step_idx]['status']
            if current_status == 'skipped':
                cp['step_records'][step_idx]['status'] = 'remedied'
                cp['step_records'][step_idx]['reason'] = '跳步后已补救'
            elif current_status == 'pending':
                # 👇 检查之前有没有触发过跳步告警，如果有，就算“补救成功”
                alarms = cp.get('jump_alarms', [])
                if any(a['current_step'] == step_idx + 1 for a in alarms):
                    cp['step_records'][step_idx]['status'] = 'remedied'
                    cp['step_records'][step_idx]['reason'] = '跳步后已补救'
                else:
                    cp['step_records'][step_idx]['status'] = 'completed'

        # 👇 核心洗白逻辑：只要步骤正常做完了，去查一下有没有坏账（跳过的步骤）

    def mark_step_skipped(self, step_idx, reason='手动跳过'):
        cp = self.current_product
        if cp and step_idx < len(cp['step_records']):
            cp['step_records'][step_idx]['status'] = 'skipped'
            cp['step_records'][step_idx]['reason'] = reason
        self._mark_ng(reason)
        
    def mark_step_remedied(self, step_idx):
        cp = self.current_product
        if cp and step_idx < len(cp['step_records']):
            cp['step_records'][step_idx]['status'] = 'remedied'
            cp['step_records'][step_idx]['reason'] = '跳步后已补救'

    def mark_step_timeout(self, step_idx, timeout_sec):
        cp = self.current_product
        if cp and step_idx < len(cp['step_records']):
            cp['step_records'][step_idx]['timeout_warning'] = True

    def get_skipped_indices(self):
        """返回当前产品中已跳过但尚未补救的步骤索引列表"""
        cp = self.current_product
        if not cp:
            return []
        return [i for i, rec in enumerate(cp['step_records']) if rec['status'] == 'skipped']

    def add_jump_alarm(self, step_idx, alarm_msg):
        cp = self.current_product
        if not cp: return
        now = datetime.now()
        if cp['jump_alarms']:
            last = cp['jump_alarms'][-1]
            h, m, s = [int(x) for x in last['time'].split(":")]
            prev_seconds = h * 3600 + m * 60 + s
            now_seconds = now.hour * 3600 + now.minute * 60 + now.second
            if (last['current_step'] == step_idx + 1 and last['msg'] == alarm_msg
                    and 0 <= now_seconds - prev_seconds < 2):
                return
        cp['jump_alarms'].append({
            'time': now.strftime("%H:%M:%S"),
            'current_step': step_idx + 1,
            'msg': alarm_msg,
        })

        # 👇 新增：只要触发了跳步告警，立刻将当前产品盖章为 NG！
        self._mark_ng("触发跳步告警")

    def on_step_advance(self, old_idx, reason='completed'):
        if reason == 'completed':
            self.mark_step_completed(old_idx)
            self._check_and_restore_ok()
        else:
            self.mark_step_skipped(old_idx, reason)

    def _check_and_restore_ok(self):
        cp = self.current_product
        if not cp or cp['status'] != 'NG': return
        has_open_issue = any(
            rec['status'] == 'skipped' or rec.get('aoi_blocked') or rec.get('aoi_forced')
            for rec in cp['step_records']
        )
        if not has_open_issue:
            cp['status'] = 'OK'
            if 'ng_reason' in cp:
                del cp['ng_reason']

    # ── 持久化与查询 ──
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.profiles_data, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        if not os.path.exists(filepath): return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 🌟 兼容老版本 JSON 数据的自动升级逻辑
            if "total_count" in data: 
                self.profiles_data = {
                    "默认方案": {
                        "ok_count": data.get("completed_cycles", 0),
                        "ng_count": data.get("ng_count", 0),
                        "history": [p for p in data.get("products", []) if p.get('status') == 'NG']
                    }
                }
            else:
                self.profiles_data = data
        except Exception:
            pass

    @property
    def ok_count(self):
        return self._get_current_db()["ok_count"]
        
    @property
    def ng_count(self):
        db = self._get_current_db()
        count = db["ng_count"]
        # 👇 实时计算：如果当前进行中的产品已经被判为 NG，立刻加到总数里！
        if self.current_product and self.current_product['status'] == 'NG':
            count += 1
        return count

    def get_ng_products(self, date_filter=None):
        db = self._get_current_db()
        ng_list = list(db["history"])
        # 👇 实时显示：把当前正在装配、且已经变成 NG 的产品也塞进记录列表中！
        if self.current_product and self.current_product['status'] == 'NG':
            ng_list.append(self.current_product)

        if date_filter:
            ng_list = [p for p in ng_list if p.get('date', '') == date_filter]
        return ng_list

    def get_available_dates(self):
        dates = sorted({p.get('date', '') for p in self._get_current_db()["history"] if p.get('date')}, reverse=True)
        return dates
