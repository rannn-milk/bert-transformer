# import library
import pandas as pd
import numpy as np
import random
import math


def readdata(df, padding_node=0, padding_correct_value=2, action_size=64, session_size=16,
             EOS_quesiton=None, EOS_skill=None, EOS_C_value=3, BOS_c_Value=4,
             session_EOS_question=None, session_EOS_skill=None, session_EOS_c_value=4,
             EOS_q_no=None, EOS_stu_no=None, SEOS_q_no=None, SEOS_stu_no=None):
    ''' create array for training

		Args:
			df(pd.dataframe)
			padding_node: padding for skill or problem
			padding_correct_value: padding value for correctness
			action_size: how many actions for each session  会话中最大操作数量
			session_size: how many sessions for each train  训练过程中考虑多少个学生会话

		Returns:
			all_train_list: the training array
	'''
    # Create training array
    all_train_list = []
    # Create val array
    all_val_list = []
    # Create Test array
    all_test_list = []

    # problem_padding_list 和 correct_padding_list 的长度均为 action_size，padding_*action_size
    problem_padding_list = [padding_node] * action_size
    correct_padding_list = [padding_correct_value] * action_size

    # 一个会话的结构
    padding_session = [problem_padding_list, problem_padding_list, correct_padding_list, problem_padding_list,
                       problem_padding_list, correct_padding_list]

    # 一个会话中问题的结束标记的列表, 创建一个包含 action_size - 1 个相同元素 session_EOS_question 的列表
    EOS_session_problem_list = [session_EOS_question] * (action_size - 1)
    EOS_session_problem_list.append(EOS_quesiton)
    EOS_session_skill_list = [session_EOS_skill] * (action_size - 1)
    EOS_session_skill_list.append(EOS_skill)
    EOS_session_correct_list = [session_EOS_c_value] * (action_size - 1)
    EOS_session_correct_list.append(EOS_C_value)
    EOS_session_qno_list = [SEOS_q_no] * (action_size - 1)
    EOS_session_qno_list.append(EOS_q_no)
    EOS_session_stuno_list = [SEOS_stu_no] * (action_size - 1)
    EOS_session_stuno_list.append(EOS_stu_no)

    EOS_session = [EOS_session_problem_list, EOS_session_skill_list, EOS_session_correct_list,
                   EOS_session_qno_list, EOS_session_stuno_list, EOS_session_correct_list]

    # 对每个学生进行处理
    for i in df.StudentID.unique():

        # 为当前学生 i 创建一个新的数据框 df_student，其中只包含该学生的所有记录
        df_student = df[df.StudentID == i]

        all_session_array = []

        # 遍历当前学生的所有唯一会话编号，目的是处理该学生每个会话的数据
        for i in range(0, len(df_student), action_size):
            # Slice data for current action_size window (session)
            df_session = df_student.iloc[i:i + action_size]

            # Create empty array for each session
            session_array = []

            # Question array
            question_array = df_session.problemId.tolist()

            # correctness_array
            correct_array = df_session.correct.tolist()

            # correctness_array
            skill_array = df_session.skill.tolist()

            # qno_array
            qd_array = df_session.Q_D.tolist()

            # stu_no_array
            stu_no_array = df_session.StudentID.tolist()

            # 如果问题数量大于或等于 action_size，则仅保留最后 action_size 个问题；否则，使用填充值填充到 action_size
            if len(question_array) >= action_size:
                question_array = question_array[(-1) * (action_size - 1):]
                skill_array = skill_array[(-1) * (action_size - 1):]
                correct_array = correct_array[(-1) * (action_size - 1):]
                qd_array = qd_array[(-1) * (action_size - 1):]
                stu_no_array = stu_no_array[(-1) * (action_size - 1):]

            else:
                # 填充 padding_node，使 question_array 达到 action_size 的长度
                question_array.extend([padding_node] * (action_size - len(question_array) - 1))
                skill_array.extend([padding_node] * (action_size - len(skill_array) - 1))
                correct_array.extend([padding_correct_value] * (action_size - len(correct_array) - 1))
                qd_array.extend([padding_node] * (action_size - len(qd_array) - 1))
                stu_no_array.extend([padding_node] * (action_size - len(stu_no_array) - 1))

            # 在问题数组末尾添加一个结束标记
            question_array.append(EOS_quesiton)
            skill_array.append(EOS_skill)
            correct_array.append(EOS_C_value)
            qd_array.append(EOS_q_no)
            stu_no_array.append(EOS_stu_no)

            enc_correct = correct_array[:-1]  # 创建一个数组，值为correct_array数组除了结束标志
            enc_correct.insert(0, BOS_c_Value)  # 在 enc_correct 的开头插入一个开始标记 BOS_c_Value，用于指示编码序列的起始

            # Append value into array for this session
            session_array.append(question_array)
            session_array.append(skill_array)
            session_array.append(correct_array)
            session_array.append(qd_array)
            session_array.append(stu_no_array)
            session_array.append(enc_correct)

            # Append value into session array for this session
            # all_session_array 现在包含了所有学生和所有会话的结构化数据
            all_session_array.append(session_array)

        all_session_array_len = len(all_session_array)
        test_session_len = all_session_array_len // 5
        train_max_session = all_session_array_len - 2 * test_session_len

        # Train array
        for ses_no in range(train_max_session - 1):

            # Create encoder input list
            one_train_list = []

            # get (num_session-1) sessions，将当前会话之前的所有会话（包括当前会话）添加到 one_train_list 中
            one_train_list.extend(all_session_array[:ses_no + 1])

            # ensure the number of session is equal with session_size
            if len(one_train_list) < session_size:

                one_train_list.extend([padding_session] * (session_size - len(one_train_list) - 1))

            else:
                one_train_list = one_train_list[(-1) * (session_size - 1):]

            one_train_list.append(EOS_session)

            # append target session，将当前会话之后的下一个会话作为目标会话添加到 one_train_list 中
            one_train_list.append(all_session_array[ses_no + 1])

            all_train_list.append(one_train_list)

        # val array
        for ses_no in range(train_max_session - 1, all_session_array_len - test_session_len - 1):

            # Create encoder input list
            one_val_list = []

            # get (num_session-1) sessions
            one_val_list.extend(all_session_array[:ses_no + 1])

            # ensure the number of session is equal with session_size
            # 如果one_train_list中的会话数量少于session_size，则通过添加填充会话（padding_session）来补齐
            # 如果会话数量超过了session_size，则只保留最后的session_size - 1个会话，确保输入数据的形状一致
            if len(one_val_list) < session_size:

                one_val_list.extend([padding_session] * (session_size - len(one_val_list) - 1))

            else:
                one_val_list = one_val_list[(-1) * (session_size - 1):]

            # 将结束标记 EOS_session 添加到训练样本中，这表明输入序列的结束
            one_val_list.append(EOS_session)

            # append target session
            one_val_list.append(all_session_array[ses_no + 1])

            all_val_list.append(one_val_list)

        # test array
        for ses_no in range(all_session_array_len - test_session_len - 1, all_session_array_len - 1):

            # Create encoder input list
            one_test_list = []

            # get (num_session-1) sessions
            one_test_list.extend(all_session_array[:ses_no + 1])

            # ensure the number of session is equal with session_size
            if len(one_test_list) < session_size:

                one_test_list.extend([padding_session] * (session_size - len(one_test_list) - 1))

            else:
                one_test_list = one_test_list[(-1) * (session_size - 1):]

            one_test_list.append(EOS_session)

            # append target session
            one_test_list.append(all_session_array[ses_no + 1])

            all_test_list.append(one_test_list)

    return all_train_list, all_val_list, all_test_list
