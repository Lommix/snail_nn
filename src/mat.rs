use std::ops;

#[macro_export]
macro_rules! mat {
    ($(($($value:expr),*)),*) => {{
        let mut data: Vec<f64> = Vec::new();
        let mut rows = 0;
        let mut cols = 0;
        $(
            let current_row = vec![$($value as f64),*];
            data.extend_from_slice(&current_row);
            rows += 1;
            cols = current_row.len();
        )*
        MatF64::new(&data, rows, cols)
    }};
}

#[derive(Clone, Debug)]
pub struct MatF64 {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl PartialEq for MatF64 {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

pub fn mat_dot(out: &mut MatF64, lhs: &MatF64, rhs: &MatF64) {
    assert_eq!(lhs.cols, rhs.rows);
    assert_eq!(out.rows, lhs.rows);
    assert_eq!(out.cols, rhs.cols);
    for r in 0..out.rows {
        for c in 0..out.cols {
            for i in 0..lhs.cols {
                out[(r, c)] += lhs[(r, i)] * rhs[(i, c)];
            }
        }
    }
}

impl MatF64 {
    pub fn new(slice: &[f64], rows: usize, cols: usize) -> MatF64 {
        MatF64 {
            data: slice.to_vec(),
            cols,
            rows,
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> MatF64 {
        MatF64 {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn rand(rows: usize, cols: usize) -> MatF64 {
        MatF64 {
            data: (0..rows * cols)
                .map(|_| (rand::random::<f64>() - 0.5) * 2.0)
                .collect(),
            rows,
            cols,
        }
    }

    pub fn empty(rows: usize, cols: usize) -> MatF64 {
        MatF64 {
            data: vec![],
            rows,
            cols,
        }
    }

    pub fn zeros_row(cols: usize) -> MatF64 {
        MatF64 {
            data: vec![0.0; cols],
            rows: 1,
            cols,
        }
    }

    pub fn clone_zero(other: &MatF64) -> MatF64 {
        MatF64 {
            data: vec![0.0; other.rows * other.cols],
            rows: other.rows,
            cols: other.cols,
        }
    }

    pub fn row_from_slice(slice: &[f64]) -> MatF64 {
        MatF64 {
            data: slice.to_vec(),
            rows: 1,
            cols: slice.len(),
        }
    }

    pub fn random_rows(cols: usize) -> MatF64 {
        MatF64 {
            data: (0..cols).map(|_| rand::random::<f64>() - 0.5).collect(),
            rows: 1,
            cols,
        }
    }

    pub fn add_row(&mut self, row: &[f64]) {
        assert!(row.len() == self.cols);
        self.rows += 1;
        self.data.extend_from_slice(row);
    }

    pub fn chunk_row(&self, chunk: usize) -> impl Iterator<Item = &[f64]> {
        (0..chunk).map(move |i| &self.data[i * self.cols..(i + 1) * self.cols])
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn split_v(self, col: usize) -> (MatF64, MatF64) {
        assert!(col < self.cols);
        assert!(col > 0);

        let mut left = MatF64::zeros(self.rows, col);
        let mut right = MatF64::zeros(self.rows, self.cols - col);

        for i in 0..self.rows {
            for j in 0..self.cols {
                if j < col {
                    left[(i, j)] = self[(i, j)];
                } else {
                    right[(i, j - col)] = self[(i, j)];
                }
            }
        }
        (left, right)
    }

    pub fn copy_from(&mut self, other: &MatF64) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.data.len() {
            self.data[i] = other.data[i];
        }
    }

    pub fn dot(&self, rhs: &MatF64) -> MatF64 {
        assert_eq!(self.cols, rhs.rows);
        let mut out = MatF64::zeros(self.rows, rhs.cols);
        mat_dot(&mut out, self, rhs);
        out
    }

    pub fn add(mut self, other: &MatF64) -> MatF64 {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
        self
    }

    pub fn transpose(&mut self) {
        self.cols = self.rows;
        self.rows = self.data.len() / self.cols;
        let size = self.data.len();
        let mut out: Vec<f64> = Vec::with_capacity(size);
        for i in 0..size {
            let row = i / self.cols; // Calculate the row index based on the current position
            let col = i % self.cols; // Calculate the column index based on the current position
            let transposed_index = col * self.rows + row;
            out.push(self.data[transposed_index % size]);
        }
        self.data = out;
    }

    pub fn get_row(&self, row: usize) -> &[f64] {
        assert!(row <= self.rows);
        &self.data[row * self.cols..(row + 1) * self.cols]
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.data.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> {
        self.data.iter_mut()
    }

    pub fn iter_rows(&self) -> impl Iterator<Item = &[f64]> {
        self.data.chunks(self.cols)
    }
}

impl ops::Index<(usize, usize)> for MatF64 {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &Self::Output {
        &self.data[r * self.cols + c]
    }
}

impl ops::IndexMut<(usize, usize)> for MatF64 {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut Self::Output {
        &mut self.data[r * self.cols + c]
    }
}

impl std::fmt::Display for MatF64 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut out = String::new();
        for r in 0..self.rows {
            if r > 0 {
                out += "\n";
            }
            out += "[";
            for c in 0..self.cols {
                out += &format!("{:.4},", self[(r, c)]);
            }
            out += "]";
        }
        write!(f, "{}", out)
    }
}

impl ops::Sub<&MatF64> for &MatF64 {
    type Output = MatF64;
    fn sub(self, other: &MatF64) -> Self::Output {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut out = MatF64::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            out.data[i] = self.data[i] - other.data[i];
        }
        out
    }
}

impl ops::Sub for MatF64 {
    type Output = MatF64;
    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut out = MatF64::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            out.data[i] = self.data[i] - other.data[i];
        }
        out
    }
}

impl ops::AddAssign for MatF64 {
    fn add_assign(&mut self, other: MatF64) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
    }
}

impl ops::AddAssign<&MatF64>for MatF64 {
    fn add_assign(&mut self, other: &MatF64) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
    }
}

impl ops::MulAssign for MatF64 {
    fn mul_assign(&mut self, other: MatF64) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.data.len() {
            self.data[i] *= other.data[i];
        }
    }
}

impl ops::MulAssign<&MatF64> for MatF64 {
    fn mul_assign(&mut self, other: &MatF64) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.data.len() {
            self.data[i] *= other.data[i];
        }
    }
}

#[test]
fn test_split() {
    let m = mat!((4, 3, 5), (1, 2, 5), (3, 4, 6));

    let (left, right) = m.split_v(2);

    let m_l = mat!((4, 3), (1, 2), (3, 4));
    let m_r = mat!((5), (5), (6));

    assert_eq!(left, m_l);
    assert_eq!(right, m_r);
}

#[test]
fn test_macro() {
    let m = mat!((4, 3, 5), (1, 2, 5), (3, 4, 6));
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 3);
}

#[test]
fn test_transpose() {
    let mut mat = mat!((6, 4, 24), (1, -9, 8));
    let out = mat!((6, 1), (4, -9), (24, 8));
    mat.transpose();
    assert_eq!(mat, out);

    let mut mat2 = mat!((6), (1), (4), (24));
    mat2.transpose();
    assert_eq!(mat2, mat!((6, 1, 4, 24)));
}

#[test]
fn test_dot() {
    let m1 = mat!((5.0, 4.0), (4.0, 6.0), (7.0, 3.0));
    let m2 = mat!((1.0, 2.0, 3.0), (4.0, 5.0, 1.0));
    let expected = mat!((21.0, 30.0, 19.0), (28.0, 38.0, 18.0), (19.0, 29.0, 24.0));
    let mut out = MatF64::zeros(3, 3);
    mat_dot(&mut out, &m1, &m2);
    assert_eq!(out, expected);
}
