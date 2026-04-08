'use client';

import React from 'react';

import { TaskDetailPageView } from './TaskDetailPageView';
import { useTaskDetailPage } from './useTaskDetailPage';

export default function TaskDetailPage(): React.ReactNode {
  const model = useTaskDetailPage();
  return <TaskDetailPageView model={model} />;
}
