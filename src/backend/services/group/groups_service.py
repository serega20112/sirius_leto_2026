class GroupService:
    def __init__(self, student_dep):
        self.student_dep = student_dep

    def get_all_groups(self) -> dict:
        students = self.student_dep.fetch_all_students()
        groups = {}
        for s in students:
            groups.setdefault(s.group_name, []).append(
                {"id": s.id, "name": s.name, "photo": f"/src/assets/images/{s.id}.jpg"}
            )
        for g in groups:
            groups[g].sort(key=lambda x: x["name"])
        return groups
