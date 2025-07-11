# Contributing to Julia110Exercises

Thank you for your interest in contributing to Julia110Exercises! This project aims to provide a comprehensive set of Julia programming exercises for learners at all levels.

## Project Goals

- Provide high-quality Julia programming exercises
- Maintain compatibility with modern Julia versions (1.10+)
- Offer clear, educational examples with multiple solution approaches
- Foster a welcoming learning environment for Julia programmers

## Ways to Contribute

### 1. Report Issues
- Found a bug? Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml)
- Have a question? Use our [question template](.github/ISSUE_TEMPLATE/question.yml)
- Want a new feature? Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml)

### 2. Improve Existing Exercises
- Fix bugs or errors in solutions
- Add alternative solution approaches
- Improve code comments and explanations
- Optimize performance
- Update for newer Julia versions

### 3. Add New Exercises
- Create exercises that fill knowledge gaps
- Focus on practical, educational problems
- Include multiple difficulty levels
- Provide clear problem statements

### 4. Improve Documentation
- Enhance README.md
- Add code comments
- Create usage examples
- Improve installation instructions

## Development Setup

### Prerequisites
- Julia 1.10 or later
- Git

### Setup Steps
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/Julia110Exercises.git
   cd Julia110Exercises
   ```
3. Install dependencies:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```
4. Test your setup:
   ```bash
   julia validate.jl
   julia test_exercises.jl
   ```

## Coding Guidelines

### Julia Style
- Follow [Julia style guidelines](https://docs.julialang.org/en/v1/manual/style-guide/)
- Use descriptive variable names
- Add helpful comments
- Keep functions focused and modular

### Code Standards
- **Compatibility**: Target Julia 1.10+
- **Performance**: Avoid common performance pitfalls
- **Readability**: Write clear, educational code
- **Testing**: Test your changes locally

### Exercise Format
Each exercise should follow this structure:
```julia
# Question XXX: [Title]
# [Clear problem description]
# 
# Example:
# Input: [example input]
# Output: [example output]

# Solution 1: [Approach description]
function solution_name_v1(args...)
    # Clear, educational implementation
    # with helpful comments
end

# Solution 2: [Alternative approach] (if applicable)
function solution_name_v2(args...)
    # Different approach or optimization
end

# Example usage:
# solution_name_v1(example_args)
```

## Testing

Before submitting a pull request:

1. **Run validation script**:
   ```bash
   julia validate.jl
   ```

2. **Run exercise tests**:
   ```bash
   julia test_exercises.jl
   ```

3. **Test specific changes**:
   ```julia
   # Test your new/modified functions
   julia> include("Julia100Exercises.jl")
   julia> your_function(test_args)
   ```

## Pull Request Process

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding guidelines

3. **Test thoroughly**:
   - Run all validation scripts
   - Test edge cases
   - Verify Julia version compatibility

4. **Commit with clear messages**:
   ```bash
   git commit -m "Add: Question XXX - Matrix operations"
   ```

5. **Push and create PR**:
   - Use our [pull request template](.github/pull_request_template.md)
   - Reference any related issues
   - Provide clear description of changes

6. **Respond to feedback** promptly and professionally

## Commit Message Guidelines

Use clear, descriptive commit messages:

- `Add: Question XXX - [Description]` - New exercises
- `Fix: Question XXX - [Issue]` - Bug fixes
- `Improve: Question XXX - [Enhancement]` - Improvements
- `Update: [Component] - [Description]` - General updates
- `Docs: [Description]` - Documentation changes

## Exercise Guidelines

### Good Exercise Characteristics
- **Educational value**: Teaches important Julia concepts
- **Clear problem statement**: Unambiguous requirements
- **Appropriate difficulty**: Matches target skill level
- **Multiple approaches**: Shows different solution strategies
- **Real-world relevance**: Practical applications when possible

### Topics to Cover
- Basic syntax and data types
- Control flow and functions
- Arrays and collections
- String manipulation
- File I/O and data processing
- Linear algebra and mathematics
- Performance optimization
- Package ecosystem usage

## Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful** and considerate in all interactions
- **Be patient** with learners asking questions
- **Provide constructive feedback** on code and ideas
- **Focus on education** and helping others learn
- **Celebrate diversity** of backgrounds and experience levels

## Getting Help

Need help contributing? Here are your options:

1. **Check existing issues** for similar questions
2. **Create a question issue** using our template
3. **Join discussions** in issue comments
4. **Review the Julia documentation** for language features

## Recognition

Contributors are recognized in several ways:

- Listed in commit history
- Mentioned in release notes for significant contributions
- Attribution in exercise comments where appropriate

## Resources

- [Julia Documentation](https://docs.julialang.org/)
- [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- [Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
- [Git Workflow Guide](https://guides.github.com/introduction/flow/)

---

Thank you for contributing to Julia110Exercises! Your efforts help make Julia more accessible to learners worldwide.
