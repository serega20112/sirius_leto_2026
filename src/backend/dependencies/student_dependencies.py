class StudentDep:
    def __init__(self, student_repo):
        self.student_repo = student_repo

    def fetch_all_students(self):
        return self.student_repo.get_all()
