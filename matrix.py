
import random
from fractions import Fraction

class Matrix:
  #Constructor
  def __init__(self, matrix=None, dims = 2):

    #Default Array
    if matrix is None:
      self.matrix = [[] for _ in range(dims)]

    elif type(matrix) == int or type(matrix) == float:
      self.matrix = [[matrix]]
    #Matrix is a 2d tensor --> Leave it at 2
    elif type(matrix) == list:
      if (self.dim(matrix)<2):
        self.matrix = [matrix]
      elif (self.correct_format(matrix)):
        self.matrix = matrix
      else:
        raise Exception("Error: Matrix must be size m x n")

    #Still gonna put this in just in case
    # elif type(matrix) == np.ndarray:
    #   matrix = matrix.tolist()
      # if (self.correct_format(matrix)):
      #   self.matrix = matrix
      # else:
      #   raise Exception("Error: Matrix must be size m x n")

    elif type(matrix) == Matrix:
      self.matrix = matrix.matrix
    #When matrix is str, int, float, etc...
    else:
      try:
        self.matrix = [i for i in matrix]
      except:
        raise Exception("Error: Data Type Not Compatible with Class \"Matrix\" ")


  def correct_format(self,m):
    #Error Handling
    flag = True
    for row in m:
      if len(row) != len(m[0]):
        flag = False
    return flag

  def dim(self,a):
    #Finds Dimension of List
    if type(a[0]) == int or type(a[0]) == float:
      return 1
    else:
      return 1 + self.dim(a[0])


  def to_float(self):
    #Converts matrix to float
    float_matrix = []
    if self.matrix is None:
      return Matrix([[] for _ in range(self.dims)])

    elif type(self.matrix) == int or type(self.matrix) == float:
      return Matrix([[float(self.matrix)]])
    #Matrix is a 2d tensor --> Leave it at 2
    elif type(self.matrix) == list:
      if (self.dim(self.matrix)<2):
        return Matrix([[float(i)] for i in self.matrix])
      elif (self.correct_format(self.matrix)):
        return Matrix([[float(item) for item in sublist] for sublist in self.matrix])

      else:
        raise Exception("Error: Cannot Convert")


  #Magic Methods
  def __call__(self):
    return self.matrix

  def __str__(self):
    final_string = ""
    for row in self.matrix:
      final_string += str(row) + "\n"
    return final_string

  #Operations
  def __add__(self, m2):
    if (self.size() != m2.size()):
      raise Exception("Error: Matrices must be the same size")

    a, b = self.size()
    final_matrix = [[None for _ in range(b)] for _ in range(a)]
    for i in range(a):
      for j in range(b):
        final_matrix[i][j] = self.matrix[i][j] + m2.matrix[i][j]
    return Matrix(final_matrix)

  def __neg__(self):
    final_matrix = self.scalar_multiply(-1)
    return Matrix(final_matrix)

  def __sub__(self, m2):
    final_matrix = self.__add__(m2.__neg__())
    return Matrix(final_matrix)

  def scalar_multiply(self, c):
    a, b = self.size()
    final_matrix = [[None for _ in range(b)] for _ in range(a)]
    for i in range(a):
      for j in range(b):
        final_matrix[i][j] = c * self.matrix[i][j]
    return Matrix(final_matrix)

  def matrix_multiply(self,m2):
    if (self.size()[1] != m2.size()[0]):
      raise Exception("Error: Incompatible Size (MxN) * (NxP) matricies required")
    a, b = self.size()
    c, d = m2.size()
    final_matrix = [[0 for _ in range(d)] for _ in range(a)]
    for i in range(a):
      for j in range(d):
        for k in range(b):
          final_matrix[i][j] += self.matrix[i][k] * m2.matrix[k][j]
    return Matrix(final_matrix)

  def __mul__(self, m2):
    if (type(m2) == Matrix):
      return self.matrix_multiply(m2)

    elif (type(m2) == list):
      return self.matrix_multiply(Matrix(m2))

  def __rmul__(self, m2):
    return self.scalar_multiply(m2)

  def __pow__(self, exp):
    if exp < 0:
      exp = abs(exp)
      inverse_matrix = self.inverse()
      for i in range(exp-1):
        inverse_matrix = inverse_matrix.matrix_multiply(inverse_matrix)
      return inverse_matrix
    else:
      final_matrix = self
      for i in range(exp-1):
        final_matrix = final_matrix.matrix_multiply(self)
      return final_matrix


  def dot(self, m2):
    return self * m2.transpose()


  #Mutators
  def set_random_matrix(self, m, n, min_val=0, max_val=9):
    #Creates a random matrix of size m,n filled with values between min_val and max_val
    self.matrix = [[random.randint(min_val, max_val) for _ in range(n)] for _ in range(m)]

  def set_matrix(self, matrix):
    #Mutator: Set matrix using a different matrix
    self.matrix = matrix

  def set_random_upper_triangle(self,s, min_val=0, max_val=9):
    #Creates a random upper triangle matrix of size m,n filled with values between min_val and max_val
    self.matrix = [[(random.randint(min_val, max_val) if i >= j else 0) for i in range(s)] for j in range(s)]

  def set_random_lower_triangle(self,s, min_val=0, max_val=9):
    #Creates a random upper triangle matrix of size m,n filled with values between min_val and max_val
    self.matrix = [[(random.randint(min_val, max_val) if i <= j else 0) for i in range(s)] for j in range(s)]

  def set_identity_matrix(self,s):
    #Creates an identity matrix of size s
    self.matrix = [[(1 if i == j else 0) for i in range(s)] for j in range(s)]

  def set_zero_matrix(self,a,b):
    #Creates a zero matrix of size s
    self.matrix = [[0 for i in range(b)] for j in range(a)]

  def set_ones_matrix(self,s):
    #Creates a ones matrix of size s
    self.matrix = [[1 for i in range(s)] for j in range(s)]

  def set_random_diagonal_matrix(self,s, min_val=0, max_val=9):
    #Creates a random diagonal matrix of size s filled with values between min_val and max_val
    self.matrix = [[(random.randint(min_val, max_val) if i == j else 0) for i in range(s)] for j in range(s)]

  def fill(self,c,s):
    if (type(c) == int):
      a,b = s,s
    else:
      a,b = s[0],s[1]

    final_matrix = [[None for _ in range(b)] for _ in range(a)]
    for i in range(a):
      for j in range(b):
        final_matrix[i][j] = c
    self.matrix = final_matrix


  def change_size(self,m,n):
    #Squishes Values into a matrix of size m,n
    a, b = self.size()
    final_matrix = [[None for _ in range(m)] for _ in range(n)]
    idx = 0
    for i in range(a):
      for j in range(b):
        final_matrix[idx//m][idx%m] = self.matrix[i][j]
        idx += 1
    self.matrix = final_matrix

  def get_column(self, col_idx):
      if type(col_idx) == int:
        if col_idx < 0 or col_idx >= len(self.matrix[0]):
            raise IndexError("Column index out of bounds")
        new_matrix = Matrix([row[col_idx] for row in self.matrix])
        return new_matrix.transpose()

      elif type(col_idx) == list:
        return Matrix([self.get_column(i) for i in col_idx])

  def get_row(self, row_idx):
      a,b = self.size()
      n = min(a,b)
      if type(row_idx) == int:
        if row_idx < 0 or row_idx >= n:
            raise IndexError("Row index out of bounds")
        return Matrix(self.matrix[row_idx])
      elif type(row_idx) == list:
        return Matrix([self.get_row(i) for i in row_idx])

  def slice_rows(self, start, end):
      return Matrix(self.matrix[start:end])

  def slice_columns(self, start, end):
      return Matrix([row[start:end] for row in self.matrix])

  def replace_row(self, row_idx, new_row):
    a,b = self.size()
    n = min(a,b)
    if row_idx < 0 or row_idx >= n:
        raise IndexError("Row index out of bounds")

    if type(new_row) == list:
        self.matrix[row_idx] = new_row
    elif type(new_row) == Matrix:
        self.matrix[row_idx] = new_row.matrix[0]

    

  def replace_col(self, col_idx, new_col):
    a,b = self.size()
    n = min(a,b)
    if col_idx < 0 or col_idx >= n:
        raise IndexError("Column index out of bounds")

    # Replace the column
    if type(new_col) == list: 
      for i in range(b):
          self.matrix[i][col_idx] = new_col[i]
    elif type(new_col) == Matrix:
      for i in range(b):
          self.matrix[i][col_idx] = new_col.matrix[i][0]

  
  #Methods
  def size(self):
    #Returns the size of matrix
    return (len(self.matrix), len(self.matrix[0]))

  def switch_rows(self,a,b):
    #Swaps rows a and b
    self.matrix[a], self.matrix[b] = self.matrix[b], self.matrix[a]


  def is_square(self):
    #Checks if matrix is square
    return self.size()[0] == self.size()[1]

  def transpose(self):
    #Returns the transpose of the matrix
    a, b = self.size()
    #Switched a and b because of transpose
    final_matrix = [[None for _ in range(a)] for _ in range(b)]
    for i in range(a):
      for j in range(b):
        final_matrix[j][i] = self.matrix[i][j]
    return Matrix(final_matrix)

  def trace(self):
    #Finds the trace of the matrix
    if (self.is_square()):
      a, b = self.size()
      total = 0
      for i in range(a):
        total += self.matrix[i][i]
      return total
    else:
      raise Exception("Error: Matrix", self.matrix ,"must be square")

  def is_lower_triangular(self):
    #Checks if matrix is lower triangular
    a, b = self.size()
    for i in range(a):
      for j in range(b):
        if (i > j and self.matrix[i][j] != 0):
          return False
    return True

  def is_upper_triangular(self):
    #Checks if matrix is upper triangular
    a, b = self.size()
    for i in range(a):
      for j in range(b):
        if (i < j and self.matrix[i][j] != 0):
          return False
    return True

  def is_diagonal(self):
    #Checks if matrix is diagonal
    a, b = self.size()
    for i in range(a):
      for j in range(b):
        if (i != j and self.matrix[i][j] != 0):
          return False
    return True

  def is_consistent(self):
    #Checks if matrix is consistent
    if self.det() == 0:
      return False
    else:
      return True

  def is_invertible(self):
    #Checks if matrix is invertible
    return self.is_square() and self.is_consistent()


    if a>b:
      return True

  def identity_matrix(self, s):
    #Creates an identity matrix of size s
    return [[(1 if i == j else 0) for j in range(s)] for i in range(s)]

  def RREF(self, right_side=1):
    #Returns the reduced row echelon form of the matrix
    a, b = self.size()
    pivot_pos = 0
    pivot = self.matrix[pivot_pos][pivot_pos]
    #Error Handling: Floating Point Arithmetic Python Error
    rref_matrix  = [[Fraction(x) for x in row] for row in self.matrix]

    # If pivot is 0, swap with a non-zero row below --> Happens only for the first calculation
    if pivot == 0:
      for i in range(pivot_pos + 1, a):
        if rref_matrix[i][pivot_pos] != 0:
          rref_matrix[pivot_pos], rref_matrix[i] = rref_matrix[i], rref_matrix[pivot_pos]
          pivot = rref_matrix[pivot_pos][pivot_pos]
          break

    for pivot_pos in range(min(a, b - right_side+1)):
      pivot = rref_matrix[pivot_pos][pivot_pos]

      # Normalize the pivot row (divide by pivot value)
      if pivot != 0:
        rref_matrix[pivot_pos] = [x / pivot for x in rref_matrix[pivot_pos]]

      # Eliminate all other entries in the pivot column
        for row in range(a):
          if row != pivot_pos:
            factor = rref_matrix[row][pivot_pos]
            rref_matrix[row] = [rref_matrix[row][i] - factor * rref_matrix[pivot_pos][i] for i in range(b)]
      else:
        rref_matrix = [[float(x) for x in row] for row in rref_matrix]
        return Matrix(rref_matrix)


    rref_matrix = [[float(x) for x in row] for row in rref_matrix]
    return Matrix(rref_matrix)

  def augmented_matrix(self, m2):
    #Returns the augmented matrix of the two matrices
    s = self.size()[0]
    a,b = self.size()
    c,d = m2.size()
    if a != c:
      raise Exception("Error: Matrices must have the same number of rows")

    augmented_matrix = [self.matrix[i] + m2.matrix[i] for i in range(s)]
    return Matrix(augmented_matrix)

  def augment_with_identity(self):
    a,b = self.size()
    s = min(a,b)
    identity = self.identity_matrix(s)
    augmented_matrix = [self.matrix[i] + identity[i] for i in range(s)]
    return Matrix(augmented_matrix)

  #Inverse does not exist if the matrix is non-square and det = 0
  def inverse(self):
    #Returns the inverse of the matrix
    if (self.is_square()):
      s = self.size()[0]
      augmented_matrix = self.augment_with_identity()
      rref_matrix = augmented_matrix.RREF(s)

      #Find if the system is consistent --> Determinant = 0
      #Det of diagonal matrix  = product of diagonal values
      for i in range(s):
        if rref_matrix.matrix[i][i] == 0:
          raise Exception("Inverse of ", self.matrix, "does not exist")

      inverse_matrix = [row[s:] for row in rref_matrix.matrix]
      return Matrix(inverse_matrix)

    else:
      raise Exception("Inverse of ", self.matrix, "does not exist")

  def bad_det(self):
    #Matrix must be square
    if (not self.is_square()):
      raise Exception("Determinant of ", self.matrix, "does not exist")

    s = self.size()[0]
    #Base Case
    if s == 1:
      return self.matrix[0][0]

    sum = 0
    upper_row = self.matrix[0]
    for i in range(s):
      top = upper_row[i]

      if top == 0:
        continue

      #Get the small matrix
      minor = []
      for row in range(1,s):
        minor.append(self.matrix[row][0:i] + self.matrix[row][i+1:])

      minor = Matrix(minor)
      cofactor = (-1) ** (i) * minor.bad_det()
      sum += top * cofactor

    return sum

  def REF_for_det(self):

    a,b = self.size()
    pivot_pos = 0

    #Scale --> For calculating determinant
    scale = 1
    pivot = self.matrix[pivot_pos][pivot_pos]


    #Error Handling: Floating Point Arithmetic Python Error
    ref_matrix  = [[Fraction(x) for x in row] for row in self.matrix]

    # If pivot is 0, swap with a non-zero row below --> Happens only for the first calculation
    if pivot == 0:
      for i in range(pivot_pos + 1, a):
        if ref_matrix[i][pivot_pos] != 0:
          scale *= -1
          ref_matrix[pivot_pos], ref_matrix[i] = ref_matrix[i], ref_matrix[pivot_pos]
          pivot = ref_matrix[pivot_pos][pivot_pos]
          break

    for pivot_pos in range(a):
      pivot = ref_matrix[pivot_pos][pivot_pos]

      # Normalize the pivot row (divide by pivot value)
      if pivot != 0:
        ref_matrix[pivot_pos] = [x / pivot for x in ref_matrix[pivot_pos]]
        scale *= pivot

      # Eliminate all other entries in the pivot column
        for row in range(pivot_pos+1,min(a, b)):
          factor = ref_matrix[row][pivot_pos]
          ref_matrix[row] = [ref_matrix[row][i] - factor * ref_matrix[pivot_pos][i] for i in range(b)]
      else:
          ref_matrix = [[float(x) for x in row] for row in ref_matrix]
          return Matrix(ref_matrix),scale


    ref_matrix = [[float(x) for x in row] for row in ref_matrix]
    return Matrix(ref_matrix), scale


  def REF(self):
    return self.REF_for_det()[0]

  def det(self):
    #Matrix must be square
    if (not self.is_square()):
      raise Exception("Determinant of ", self.matrix, "does not exist")

    return self.REF_for_det()[1]


  def qr_decomposition(self):
    """
    Performs QR decomposition using the Gram-Schmidt process.
    Returns matrices Q and R such that A = Q * R.
    """
    n_rows, n_cols = self.size()
    
    Q = [[0] * n_cols for _ in range(n_rows)]  # Initialize Q
    R = [[0] * n_cols for _ in range(n_cols)]  # Initialize R
    
    for k in range(n_cols):
        # Copy the k-th column of A to start the process
        Q_k = [row[k] for row in self.matrix]
        
        # Orthogonalize against previous Q columns
        for j in range(k):
            Q_j = [row[j] for row in Q]
            R[j][k] = self.dot_product_vector(Q_j, Q_k)  # Compute R[j, k]
            Q_k = self.vector_subtract(Q_k, self.scalar_multiply_vector(Q_j, R[j][k]))
        
        # Compute the norm for Q_k and normalize
        R[k][k] = self.vector_norm(Q_k)
        Q_k = self.scalar_multiply_vector(Q_k, 1 / R[k][k])
        
        # Place the normalized vector into the k-th column of Q
        for i in range(n_rows):
            Q[i][k] = Q_k[i]
    
    return Matrix(Q), Matrix(R)

  def make_similar(self):
    Q,R = self.qr_decomposition()
    B = R.dot(Q)
    return B

  def eigenvalues(self, tolerance = 1e-10):
    B = self.make_similar()
    iters = 0
    leig = B.matrix[-1][-1]
    diff = 1
    while diff> tolerance:
      B = B.make_similar()
      iters += 1
      diff = abs(leig - B.matrix[-1][-1])
      leig = B.matrix[-1][-1]
    eigs = [B.matrix[i][i] for i in range(len(B.matrix))]

    return eigs


  def eigenvectors(self):
      """
      Computes the eigenvectors of the matrix by solving the null space of (A - λI).
      """
      all_eigenvectors = []
      all_eigenvalues = self.eigenvalues()
      a, b = self.size()
      
      I = Matrix()
      I.set_identity_matrix(b)  # Identity matrix of appropriate size

      for eigenvalue in all_eigenvalues:
          # Characteristic Equation: C = A - λI
          C = self - (eigenvalue * I)
          C = C.RREF()  # Reduce the matrix to RREF to solve for null space

          rows, cols = C.size()

          # Step 1: Identify pivot columns
          pivot_columns = []
          for row in range(rows):
              for col in range(cols):
                  if C.matrix[row][col] != 0:
                      pivot_columns.append(col)
                      break

          # Step 2: Identify free variables
          free_variables = [col for col in range(cols) if col not in pivot_columns]

          # Step 3: Solve for the null space
          for free_var in free_variables:
              vector = [0] * cols
              vector[free_var] = 1  # Assign 1 to the free variable
              
              # Solve for pivot variables by back substitution
              for row in range(rows - 1, -1, -1):  # Iterate over rows backward
                  pivot_col = -1
                  for col in range(cols):
                      if C.matrix[row][col] != 0:  # Find the pivot column
                          pivot_col = col
                          break
                  
                  if pivot_col != -1:  # Solve for pivot variable
                      vector[pivot_col] = -sum(
                          C.matrix[row][col] * vector[col] for col in range(pivot_col + 1, cols)
                      )

              # Append the eigenvector (as a column vector)
              eigenvector = Matrix([vector]).transpose()
              all_eigenvectors.append(eigenvector)

      return all_eigenvectors



  def eigen_results(self):
    eigenvalue_matrix = self.eigenvalues()
    eigenvalue_matrix = Matrix(eigenvalue_matrix)
    eigenvalue_matrix = eigenvalue_matrix.transpose()



    all_eigenvectors = self.eigenvectors()
    if len(all_eigenvectors) == 0:
      return eigenvalue_matrix,Matrix()
    original = all_eigenvectors[0]
    for i in range(1, len(all_eigenvectors)):
      original = original.augmented_matrix(all_eigenvectors[i])

    eigenvector_matrix = original
    return eigenvalue_matrix,eigenvector_matrix

  def dot_product_vector(self, vec1, vec2):
    """Calculates the dot product of two vectors."""
    return sum(x * y for x, y in zip(vec1, vec2))

  def vector_norm(self, vec):
      """Calculates the Euclidean norm of a vector."""
      return sum(x**2 for x in vec) ** 0.5

  def scalar_multiply_vector(self, vec, scalar):
      """Multiplies a vector by a scalar."""
      return [x * scalar for x in vec]

  def vector_subtract(self, vec1, vec2):
      """Subtracts one vector from another."""
      return [x - y for x, y in zip(vec1, vec2)]





